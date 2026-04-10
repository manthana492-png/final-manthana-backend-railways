# Deploy all Manthana Modal apps (run from repo with Modal CLI authenticated).
# Usage: cd manthana-backend; .\scripts\modal_deploy_all.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$deploys = @(
    "modal_app/deploy_ct_brain.py",
    "modal_app/deploy_brain_mri.py",
    "modal_app/deploy_cardiac_ct.py",
    "modal_app/deploy_spine_neuro.py",
    "modal_app/deploy_abdominal_ct.py",
    "modal_app/deploy_body_xray.py",
    "modal_app/deploy_ultrasound.py",
    "modal_app/deploy_pathology.py",
    "modal_app/deploy_cytology.py",
    "modal_app/deploy_mammography.py",
    "modal_app/deploy_lab_report.py",
    "modal_app/deploy_oral_cancer.py",
    "modal_app/deploy_ecg.py",
    "modal_app/deploy_dermatology.py",
    "modal_app/deploy_ct_brain_vista.py"
)

foreach ($d in $deploys) {
    Write-Host "=== modal deploy $d ===" -ForegroundColor Cyan
    modal deploy $d
}

Write-Host "Done. Set Railway *_SERVICE_URL and CT_BRAIN_VISTA_SERVICE_URL from Modal dashboard URLs." -ForegroundColor Green
