from __future__ import annotations

from io import BytesIO

from minio import Minio

from rag_ddd.domain.ports import BlobStore


class MinIOBlobStore(BlobStore):
    """Self-hosted S3-compatible blob storage via MinIO.

    All data stays on-premise — no cloud dependency.
    """

    def __init__(
        self,
        url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        # Minio client expects host:port without scheme
        endpoint = url.replace("http://", "").replace("https://", "")
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def put(self, path: str, content: bytes, content_type: str | None = None) -> str:
        data = BytesIO(content)
        self._client.put_object(
            self._bucket,
            path,
            data,
            length=len(content),
            content_type=content_type or "application/octet-stream",
        )
        return path

    def get(self, path: str) -> bytes:
        response = self._client.get_object(self._bucket, path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete(self, path: str) -> None:
        self._client.remove_object(self._bucket, path)
