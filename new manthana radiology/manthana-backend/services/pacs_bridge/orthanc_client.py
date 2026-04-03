"""
Orthanc REST API Client — async httpx wrapper.
"""
import httpx
from typing import Optional, Any


class OrthancClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                auth=self.auth,
                timeout=60.0,
            )
        return self._client

    async def check_health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get("/system")
            return resp.status_code == 200
        except Exception:
            return False

    async def get_json(self, path: str) -> Any:
        client = await self._get_client()
        resp = await client.get(path)
        resp.raise_for_status()
        return resp.json()

    async def get_bytes(self, path: str) -> bytes:
        client = await self._get_client()
        resp = await client.get(path)
        resp.raise_for_status()
        return resp.content

    async def post_json(self, path: str, data: Any) -> Any:
        client = await self._get_client()
        resp = await client.post(path, json=data)
        resp.raise_for_status()
        return resp.json()

    async def put_json(self, path: str, data: Any) -> Any:
        client = await self._get_client()
        resp = await client.put(path, json=data)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def put_metadata(self, path: str, value: str):
        client = await self._get_client()
        resp = await client.put(path, content=value, headers={"Content-Type": "text/plain"})
        resp.raise_for_status()

    async def get_metadata(self, path: str) -> Optional[str]:
        try:
            client = await self._get_client()
            resp = await client.get(path)
            if resp.status_code == 200:
                return resp.text.strip().strip('"')
            return None
        except Exception:
            return None

    async def delete(self, path: str):
        client = await self._get_client()
        resp = await client.delete(path)
        resp.raise_for_status()

    async def wado_rs(self, path: str) -> Any:
        """Proxy DICOMweb WADO-RS request."""
        client = await self._get_client()
        resp = await client.get(f"/dicom-web{path}")
        resp.raise_for_status()
        return resp.json()

    async def list_studies(self, limit: int = 50, offset: int = 0) -> list:
        """List studies with expand=true for full details."""
        client = await self._get_client()
        resp = await client.get(f"/studies?expand&limit={limit}&since={offset}")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
