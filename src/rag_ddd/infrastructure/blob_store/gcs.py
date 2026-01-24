from __future__ import annotations

from google.cloud import storage

from rag_ddd.domain.ports import BlobStore


class GCSBlobStore(BlobStore):
    def __init__(self, bucket: str) -> None:
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    def put(self, path: str, content: bytes, content_type: str | None = None) -> str:
        blob = self._bucket.blob(path)
        blob.upload_from_string(content, content_type=content_type)
        return blob.name

    def get(self, path: str) -> bytes:
        blob = self._bucket.blob(path)
        return blob.download_as_bytes()

    def delete(self, path: str) -> None:
        blob = self._bucket.blob(path)
        blob.delete()
