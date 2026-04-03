"""
Worklist Manager — CRUD operations for DICOM worklist via Orthanc REST API.
"""
import uuid
import json
from typing import List, Dict, Any, Optional
from orthanc_client import OrthancClient


class WorklistManager:
    def __init__(self, orthanc: OrthancClient):
        self.orthanc = orthanc
        self._cache: Dict[str, dict] = {}

    async def list_items(self) -> List[dict]:
        """List all worklist items from Orthanc."""
        try:
            # Try the new Worklists REST API (Orthanc 2025+)
            items = await self.orthanc.get_json("/worklists")
            result = []
            for item_id in items:
                try:
                    detail = await self.orthanc.get_json(f"/worklists/{item_id}")
                    detail["id"] = item_id
                    result.append(detail)
                except Exception:
                    result.append({"id": item_id})
            return result
        except Exception:
            # Fallback: return cached items (for older Orthanc)
            return list(self._cache.values())

    async def create_item(self, data: dict) -> dict:
        """Create a new worklist item."""
        item_id = data.get("id") or str(uuid.uuid4())[:8]

        # Build DICOM worklist tags
        worklist_tags = {
            "PatientName": data.get("patient_name", ""),
            "PatientID": data.get("patient_id", ""),
            "AccessionNumber": data.get("accession_number", item_id),
            "ScheduledProcedureStepSequence": [{
                "Modality": data.get("modality", "CR"),
                "ScheduledStationAETitle": data.get("scheduled_aet", "MANTHANA"),
                "ScheduledProcedureStepStartDate": data.get("scheduled_date", ""),
                "ScheduledProcedureStepStartTime": data.get("scheduled_time", ""),
                "ScheduledProcedureStepDescription": data.get("procedure_description", ""),
                "ScheduledPerformingPhysicianName": data.get("referring_physician", ""),
            }],
        }

        try:
            # Try Orthanc Worklists REST API
            result = await self.orthanc.post_json("/worklists", worklist_tags)
            data["id"] = result.get("ID", item_id)
        except Exception:
            # Fallback: cache locally
            data["id"] = item_id

        self._cache[data["id"]] = data
        return data

    async def update_item(self, item_id: str, data: dict) -> dict:
        """Update an existing worklist item."""
        data["id"] = item_id
        self._cache[item_id] = data

        try:
            worklist_tags = {
                "PatientName": data.get("patient_name", ""),
                "PatientID": data.get("patient_id", ""),
                "AccessionNumber": data.get("accession_number", item_id),
            }
            await self.orthanc.put_json(f"/worklists/{item_id}", worklist_tags)
        except Exception:
            pass

        return data

    async def delete_item(self, item_id: str) -> dict:
        """Delete a worklist item."""
        self._cache.pop(item_id, None)
        try:
            await self.orthanc.delete(f"/worklists/{item_id}")
        except Exception:
            pass
        return {"status": "deleted", "id": item_id}
