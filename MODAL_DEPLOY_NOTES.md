# Modal Deploy Notes

## Correct Deploy Path
Always deploy from `this_studio`, NOT `_railway_upload`:
```
cd "D:\studio-backup\this_studio\new manthana radiology\manthana-backend"
python -m modal deploy modal_app/deploy_<service>.py
```

## Why _railway_upload Fails
`_railway_upload` is a git mirror only — it has no `packages/` folder.
Deploying from there fails with:
```
RuntimeError: manthana-inference not found at D:\studio-backup\_railway_upload\packages\manthana-inference
```

## Successful Build Note (2026-04-14)
- Built image `im-zk9V7ehmSIYKvUX1IJCVCv` in 5.10s
- `deploy_body_xray` deployed successfully:
  - Endpoint: https://manthana492-prod-2--manthana-body-xray-serve.modal.run
  - Dashboard: https://modal.com/apps/manthana492-prod-2/main/deployed/manthana-body-xray

## Windows Encoding Fix
If you get `charmap codec can't encode` errors, set UTF-8 first:
```powershell
$env:PYTHONIOENCODING="utf-8"
python -m modal deploy modal_app/deploy_<service>.py
```
