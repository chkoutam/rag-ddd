from __future__ import annotations

from typing import Any, Mapping

from bson import ObjectId
from pymongo import MongoClient

from rag_ddd.domain.ports import DocumentStore


class MongoDocumentStore(DocumentStore):
    def __init__(self, uri: str, database: str, collection: str = "documents") -> None:
        self._client = MongoClient(uri)
        self._collection = self._client[database][collection]
        self._collection.create_index("checksum", unique=True, sparse=True)

    def create(self, document: Mapping[str, Any]) -> str:
        result = self._collection.insert_one(dict(document))
        return str(result.inserted_id)

    def update(self, doc_id: str, patch: Mapping[str, Any]) -> None:
        self._collection.update_one({"_id": ObjectId(doc_id)}, {"$set": dict(patch)})

    def get(self, doc_id: str) -> Mapping[str, Any] | None:
        doc = self._collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return None
        return self._serialize(doc)

    def list(self, filters: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]:
        cursor = self._collection.find(dict(filters) if filters else {})
        return [self._serialize(doc) for doc in cursor]

    def delete(self, doc_id: str) -> None:
        self._collection.delete_one({"_id": ObjectId(doc_id)})

    @staticmethod
    def _serialize(document: Mapping[str, Any]) -> Mapping[str, Any]:
        data = dict(document)
        if "_id" in data:
            data["_id"] = str(data["_id"])
        return data
