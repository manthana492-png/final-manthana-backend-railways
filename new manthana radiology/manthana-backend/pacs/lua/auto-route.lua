-- ══════════════════════════════════════════════════════════
-- Manthana PACS — Auto-Route Lua Script
-- 
-- Triggered on stable studies. Auto-forwards incoming DICOM
-- to the Manthana Gateway for AI analysis.
-- ══════════════════════════════════════════════════════════

-- Configuration
local GATEWAY_URL = "http://gateway:8000"
local BRIDGE_URL  = "http://pacs_bridge:8030"
local AUTO_ROUTE_ENABLED = true

-- ── Modality detection from DICOM tags ──
function DetectModality(tags)
    local modality = tags["0008,0060"] or ""        -- Modality tag
    local bodyPart = tags["0018,0015"] or ""        -- Body Part Examined
    local studyDesc = tags["0008,1030"] or ""       -- Study Description
    local seriesDesc = tags["0008,103e"] or ""      -- Series Description

    modality = string.upper(modality)
    bodyPart = string.upper(bodyPart)
    studyDesc = string.upper(studyDesc)

    -- Map DICOM modality + body part → Manthana service
    if modality == "CR" or modality == "DX" then
        return "xray"
    elseif modality == "MR" or modality == "MRI" then
        if string.find(bodyPart, "BRAIN") or string.find(bodyPart, "HEAD") then
            return "brain_mri"
        elseif string.find(bodyPart, "SPINE") then
            return "spine_neuro"
        end
        return "brain_mri"  -- default MRI → brain
    elseif modality == "CT" then
        if string.find(bodyPart, "CHEST") or string.find(bodyPart, "HEART") then
            return "cardiac_ct"
        elseif string.find(bodyPart, "ABDOMEN") or string.find(bodyPart, "PELVIS") then
            return "abdominal_ct"
        elseif string.find(bodyPart, "SPINE") then
            return "spine_neuro"
        elseif string.find(bodyPart, "HEAD") or string.find(bodyPart, "BRAIN") then
            return "brain_mri"  -- CT head uses brain service
        end
        return "abdominal_ct"  -- default CT → abdominal
    elseif modality == "US" then
        return "ultrasound"
    elseif modality == "MG" then
        return "mammography"
    elseif modality == "PT" or modality == "NM" then
        return "xray"  -- PET/Nuclear → fallback
    elseif modality == "OPG" or modality == "IO" or modality == "PX" then
        return "xray"
    elseif modality == "ECG" or modality == "HD" then
        return "ecg"
    end

    -- Fallback: try study description
    if string.find(studyDesc, "CHEST") or string.find(studyDesc, "XRAY") then
        return "xray"
    end

    return "xray"  -- ultimate fallback
end


-- ══════════════════════════════════════════════════════════
-- OnStableStudy — triggered when a study hasn't received
-- new instances for StableAge seconds (default: 60s)
-- ══════════════════════════════════════════════════════════
function OnStableStudy(studyId, tags, metadata)
    if not AUTO_ROUTE_ENABLED then
        PrintToLog("Auto-route disabled, skipping study: " .. studyId)
        return
    end

    PrintToLog("═══ MANTHANA AUTO-ROUTE ═══")
    PrintToLog("Stable study received: " .. studyId)

    -- Get study details
    local study = ParseJson(RestApiGet("/studies/" .. studyId))
    local patientName = study.PatientMainDicomTags.PatientName or "ANONYMOUS"
    local patientId = study.PatientMainDicomTags.PatientID or ""
    local modality = DetectModality(tags)

    PrintToLog("Patient: " .. patientName .. " | Modality: " .. modality)

    -- Get the first series and first instance
    local series = ParseJson(RestApiGet("/studies/" .. studyId .. "/series"))
    if #series == 0 then
        PrintToLog("ERROR: No series found in study " .. studyId)
        return
    end

    local firstSeries = series[1]
    local instances = ParseJson(RestApiGet("/series/" .. firstSeries.ID .. "/instances"))
    if #instances == 0 then
        PrintToLog("ERROR: No instances found in series")
        return
    end

    -- Notify the PACS Bridge to process this study
    local payload = {
        study_id = studyId,
        patient_name = patientName,
        patient_id = patientId,
        modality = modality,
        series_count = #series,
        instance_count = 0
    }

    -- Count total instances
    for _, s in ipairs(series) do
        local si = ParseJson(RestApiGet("/series/" .. s.ID .. "/instances"))
        payload.instance_count = payload.instance_count + #si
    end

    PrintToLog("Sending to PACS Bridge: " .. modality .. " (" .. payload.instance_count .. " instances)")

    -- Tag the study with pending AI status
    RestApiPut("/studies/" .. studyId .. "/metadata/ai-status", "pending")

    -- Notify bridge (fire-and-forget via Orthanc job)
    local httpHeaders = {}
    httpHeaders["Content-Type"] = "application/json"

    local success, err = pcall(function()
        HttpPost(BRIDGE_URL .. "/pacs/auto-route", DumpJson(payload), httpHeaders)
    end)

    if success then
        PrintToLog("✓ Auto-route notification sent for: " .. patientName)
        RestApiPut("/studies/" .. studyId .. "/metadata/ai-status", "queued")
    else
        PrintToLog("✗ Auto-route notification failed: " .. tostring(err))
        RestApiPut("/studies/" .. studyId .. "/metadata/ai-status", "route-failed")
    end
end


-- ══════════════════════════════════════════════════════════
-- OnStoredInstance — triggered for each individual instance
-- Lightweight: just logs, doesn't trigger analysis
-- ══════════════════════════════════════════════════════════
function OnStoredInstance(instanceId, tags, metadata, origin)
    -- Only log remote C-STORE (not REST uploads during analysis)
    if origin.RequestOrigin == "DicomProtocol" then
        local aet = origin.RemoteAet or "UNKNOWN"
        local modality = tags["0008,0060"] or "??"
        PrintToLog("C-STORE received: " .. modality .. " from " .. aet)
    end
end


-- ══════════════════════════════════════════════════════════
-- Log startup
-- ══════════════════════════════════════════════════════════
PrintToLog("═══════════════════════════════════════")
PrintToLog("  Manthana PACS Auto-Route Script v1.0")
PrintToLog("  Auto-route: " .. tostring(AUTO_ROUTE_ENABLED))
PrintToLog("  Gateway: " .. GATEWAY_URL)
PrintToLog("  Bridge: " .. BRIDGE_URL)
PrintToLog("═══════════════════════════════════════")
