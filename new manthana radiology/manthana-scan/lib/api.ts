/* ═══ API Client — Gateway Integration ═══ */
import { API_BASE, GATEWAY_URL } from "./constants";
import type {
  AnalysisResponse,
  JobStatus,
  ServiceHealth,
  UnifiedAnalysisResult,
} from "./types";
import { AnalysisCancelledError } from "./errors";

const FRONTEND_TO_BACKEND_MODALITY: Record<string, string> = {
  xray: "xray",
  /** @deprecated Prefer ct_abdomen / ct_chest / ct_cardiac / ct_spine / ct_brain */
  ct: "ct",
  ct_abdomen: "abdominal_ct",
  ct_chest: "chest_ct",
  ct_cardiac: "cardiac_ct",
  ct_spine: "spine_neuro",
  ct_brain: "ct_brain",
  brain_mri: "brain_mri",
  /** Legacy; gateway resolves to brain_mri (not spine). */
  mri: "mri",
  spine_mri: "spine_mri",
  mr_spine: "spine_mri",
  ultrasound: "ultrasound",
  ecg: "ecg",
  pathology: "pathology",
  mammography: "mammography",
  cytology: "cytology",
  oral_cancer: "oral_cancer",
  lab_report: "lab_report",
  dermatology: "dermatology",
  // CT sub-modalities from CtWizardState
  abdominal_ct: "abdominal_ct",
  chest_ct: "chest_ct",
  cardiac_ct: "cardiac_ct",
  spine_neuro: "spine_neuro",
  brain_ct: "ct_brain",
  head_ct: "ct_brain",
  ncct_brain: "ct_brain",
};

function normalizeResult(data: unknown): AnalysisResponse {
  const r = data as AnalysisResponse;

  // 1. Normalize per-finding confidence (backend may return 0–100)
  if (Array.isArray(r.findings)) {
    r.findings = r.findings.map((f) => ({
      ...f,
      confidence:
        typeof f.confidence === "number" && f.confidence > 1
          ? f.confidence / 100
          : f.confidence,
    }));
  }

  // 2. Normalize top-level confidence if backend ever returns it as number
  // (current backend uses string, so this is defensive only).
  if (typeof r.confidence === "number" && r.confidence > 1) {
    r.confidence = r.confidence / 100;
  }

  // 3. Resolve relative heatmap URLs (top-level)
  if (typeof r.heatmap_url === "string" && r.heatmap_url.startsWith("/")) {
    r.heatmap_url = `${GATEWAY_URL}${r.heatmap_url}`;
  }

  // 4. Resolve relative heatmap URLs on findings, if any
  if (Array.isArray(r.findings)) {
    r.findings = r.findings.map((f) => ({
      ...f,
      heatmap_url:
        typeof f.heatmap_url === "string" && f.heatmap_url.startsWith("/")
          ? `${GATEWAY_URL}${f.heatmap_url}`
          : f.heatmap_url,
    }));
  }

  return r;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new AnalysisCancelledError());
    const id = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener(
        "abort",
        () => {
          clearTimeout(id);
          reject(new AnalysisCancelledError());
        },
        { once: true }
      );
    }
  });
}

async function pollJobUntilComplete(
  jobId: string,
  signal?: AbortSignal
): Promise<AnalysisResponse> {
  const MAX_POLLS = 60;
  const POLL_INTERVAL_MS = 2000;

  for (let i = 0; i < MAX_POLLS; i++) {
    if (signal?.aborted) {
      throw new AnalysisCancelledError();
    }

    const status = await getJobStatus(jobId, signal);

    if (status.status === "queued" || status.status === "processing") {
      await sleep(POLL_INTERVAL_MS, signal);
      continue;
    }

    if (status.status === "complete") {
      if (!status.result) {
        throw new Error(`Job ${jobId} complete but result is missing.`);
      }
      return normalizeResult(status.result);
    }

    if (status.status === "failed") {
      throw new Error(status.error ?? `Job ${jobId} failed with no error detail.`);
    }

    throw new Error(`Job ${jobId} returned unexpected status: "${status.status}"`);
  }

  throw new Error(
    `Analysis timed out after ${(MAX_POLLS * POLL_INTERVAL_MS) / 1000}s.`
  );
}

async function readGatewayError(res: Response): Promise<string> {
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) {
    try {
      const data = (await res.json()) as { detail?: unknown };
      const d = data?.detail;
      if (typeof d === "string") return d;
      if (Array.isArray(d)) return d.map((x) => (typeof x === "object" && x && "msg" in x ? String((x as { msg: string }).msg) : String(x))).join("; ");
      if (d != null) return JSON.stringify(d);
    } catch {
      /* fall through */
    }
  }
  return res.text().catch(() => `HTTP ${res.status}`);
}

/** Analyse a medical image — main entry point (requires Bearer JWT: set via setGatewayAuthToken / login). */
export async function analyzeImage(
  file: File,
  modality: string,
  patientId?: string,
  clinicalNotes?: string,
  patientContext?: Record<string, unknown>,
  signal?: AbortSignal
): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("file", file);

  const backendModality = FRONTEND_TO_BACKEND_MODALITY[modality];
  if (!backendModality) {
    // Surface unsupported modality early instead of sending a bad request
    console.warn(
      `[analyzeImage] Unknown frontend modality "${modality}" — no backend mapping found.`
    );
    throw new Error(`Unsupported modality: ${modality}`);
  }
  form.append("modality", backendModality);
  if (patientId) form.append("patient_id", patientId);
  if (clinicalNotes) form.append("clinical_notes", clinicalNotes);
  if (patientContext && Object.keys(patientContext).length > 0) {
    form.append("patient_context_json", JSON.stringify(patientContext));
  }

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    // Authorization is added in the Next.js route handler proxy
    body: form,
    signal,
  });

  if (res.status === 401) {
    throw new Error("Authentication required. Please log in.");
  }
  if (!res.ok) {
    const err = await readGatewayError(res);
    throw new Error(`Analysis failed (${res.status}): ${err}`);
  }

  const data = await res.json();

  // Gateway encodes async via JSON body status (not HTTP 202)
  if (data.status === "queued" && data.job_id) {
    return pollJobUntilComplete(data.job_id, signal);
  }

  // Sync / triage fast-path
  // "triage" included defensively; gateway currently normalises triage → "complete"
  // but this guard keeps us forward-compatible if that changes.
  if (!data.status || data.status === "complete" || data.status === "triage") {
    return normalizeResult(data);
  }

  throw new Error(
    `Unexpected analysis status "${data.status}" for job ${data.job_id ?? "unknown"}`
  );
}

/** Poll async job status */
export async function getJobStatus(
  jobId: string,
  signal?: AbortSignal
): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/job/${jobId}/status`, {
    signal,
  });
  if (res.status === 401) {
    throw new Error("Authentication required. Please log in.");
  }
  if (!res.ok) throw new Error(`Job status failed: ${res.status}`);
  return res.json();
}

/** Get all service health statuses */
export async function getServicesHealth(): Promise<ServiceHealth[]> {
  try {
    const res = await fetch(`${API_BASE}/health/services`);
    if (!res.ok) return [];

    const data = await res.json();
    const services = Array.isArray(data?.services) ? data.services : [];

    return services.map(
      (s: { modality?: string; status?: string }): ServiceHealth => {
        const modality = s.modality ?? "unknown";
        return {
          id: modality,
          name: modalityDisplayName(modality),
          status: normalizeServiceStatus(s.status),
          latency_ms: null,
        };
      }
    );
  } catch {
    // Never throw from health polling; callers can treat empty as "unknown"
    return [];
  }
}

/** Get gateway health */
export async function getGatewayHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Generate narrative report (proxies to report_assembly via gateway POST /report). */
export async function generateReport(
  analysisResult: AnalysisResponse,
  language: string = "en"
): Promise<{ report: string; pdf_url?: string; impression?: string; narrative?: string }> {
  const res = await fetch(`${API_BASE}/report`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ...analysisResult, language }),
  });
  if (res.status === 401) {
    throw new Error("Authentication required. Please log in.");
  }
  if (!res.ok) throw new Error(`Report generation failed: ${res.status}`);
  const data = (await res.json()) as {
    narrative?: string;
    impression?: string;
    pdf_url?: string;
  };
  return {
    report: data.narrative ?? "",
    pdf_url: data.pdf_url,
    narrative: data.narrative,
    impression: data.impression,
  };
}

/** Ask AI co-pilot a question about findings */
export async function askCoPilot(
  question: string,
  context?: Record<string, unknown>
): Promise<string> {
  try {
    const res = await fetch(`${API_BASE}/copilot`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, context }),
    });

    // Endpoint not built yet — return friendly placeholder
    if (res.status === 404) {
      return "AI Co-Pilot is coming soon. This feature will be available in a future update.";
    }

    if (!res.ok) {
      const msg = await readGatewayError(res);
      throw new Error(`Co-Pilot service error (${res.status}): ${msg}`);
    }

    const data = (await res.json()) as { response?: string };
    if (typeof data.response === "string" && data.response.trim().length > 0) {
      return data.response;
    }

    // Defensive fallback if backend shape changes
    return "The AI Co-Pilot did not return a response. Please try again later.";
  } catch (err) {
    // Network or other unexpected failures → user-facing unavailability message
    console.error("[askCoPilot] Failed to reach Co-Pilot service:", err);
    return "AI Co-Pilot is temporarily unavailable. Please check your connection or try again later.";
  }
}

// Maps backend modality string to a human-readable display name
function modalityDisplayName(modality: string): string {
  const NAMES: Record<string, string> = {
    xray: "X-Ray",
    ct: "CT Scan",
    mri: "MRI",
    brain_mri: "Brain MRI",
    spine_mri: "Spine MRI",
    spine_neuro: "Spine / Neuro",
    mammography: "Mammography",
    ecg: "ECG",
    dermatology: "Dermatology",
    oral_cancer: "Oral Cancer",
    pathology: "Pathology",
    lab_report: "Lab Report",
    cytology: "Cytology",
    ultrasound: "Ultrasound",
  };
  return NAMES[modality] ?? modality;
}

// Coerces backend status strings to the union the UI expects
function normalizeServiceStatus(
  status: string | undefined
): "online" | "offline" | "degraded" | "unknown" {
  if (status === "online" || status === "offline" || status === "degraded") {
    return status;
  }
  return "unknown";
}

/** Request unified cross-modality report from report_assembly via gateway POST /unified-report */
export async function requestUnifiedReport(
  individualResults: { modality: string; result: AnalysisResponse }[],
  patientId: string,
  language: string = "en"
): Promise<UnifiedAnalysisResult> {
  const res = await fetch(`${API_BASE}/unified-report`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ results: individualResults, patient_id: patientId, language }),
  });
  if (res.status === 401) {
    throw new Error("Authentication required. Please log in.");
  }
  if (!res.ok) {
    const err = await res.text().catch(() => "Unknown error");
    throw new Error(`Unified report failed (${res.status}): ${err}`);
  }
  return res.json();
}

/* ═══ PACS API ═══ */

import type { PacsStudy, WorklistItem, PacsConfig } from "./types";

const PACS_BASE = `${API_BASE}/pacs`;

/** Fetch studies from Orthanc PACS */
export async function fetchPacsStudies(filters?: {
  patient_name?: string;
  patient_id?: string;
  modality?: string;
  date_from?: string;
  date_to?: string;
  limit?: number;
}): Promise<PacsStudy[]> {
  const params = new URLSearchParams();
  if (filters?.patient_name) params.set("patient_name", filters.patient_name);
  if (filters?.patient_id) params.set("patient_id", filters.patient_id);
  if (filters?.modality) params.set("modality", filters.modality);
  if (filters?.date_from) params.set("date_from", filters.date_from);
  if (filters?.date_to) params.set("date_to", filters.date_to);
  if (filters?.limit) params.set("limit", String(filters.limit));

  const url = `${PACS_BASE}/studies${params.toString() ? "?" + params : ""}`;
  const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
  if (!res.ok) return [];
  return res.json();
}

/** Fetch worklist items */
export async function fetchWorklist(): Promise<WorklistItem[]> {
  try {
    const res = await fetch(`${PACS_BASE}/worklist`, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

/** Create worklist item */
export async function createWorklistItem(item: Omit<WorklistItem, "id">): Promise<WorklistItem> {
  const res = await fetch(`${PACS_BASE}/worklist`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(item),
  });
  if (!res.ok) throw new Error("Failed to create worklist item");
  return res.json();
}

/** Send a PACS study to AI analysis */
export async function sendStudyToAI(
  studyId: string,
  modalityOverride?: string
): Promise<{ status: string; job_id: string }> {
  const res = await fetch(`${PACS_BASE}/send-to-ai`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ study_id: studyId, modality_override: modalityOverride }),
  });
  if (!res.ok) throw new Error("Failed to send study to AI");
  return res.json();
}

/** Test PACS connectivity (C-ECHO) */
export async function echoPacs(modalityName: string): Promise<{ status: string }> {
  const res = await fetch(`${PACS_BASE}/modalities/${modalityName}/echo`, {
    method: "POST",
    signal: AbortSignal.timeout(10000),
  });
  return res.json();
}

/** Get PACS config */
export async function getPacsConfig(): Promise<PacsConfig | null> {
  try {
    const res = await fetch(`${PACS_BASE}/config`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
