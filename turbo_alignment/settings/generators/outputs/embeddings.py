from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmbeddingsInferenceOutput(BaseModel):
    id: str
    dataset_name: str
    embedding: List[float]
    item_ids: List[str]
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ItemEmbeddingOutput(BaseModel):
    id: str
    item_id: str
    dataset_name: str
    embedding: List[float]
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)
    item_text: str = Field(default_factory=str)


class ItemEmbedding(BaseModel):
    id: str
    embedding: List[float]
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict) 