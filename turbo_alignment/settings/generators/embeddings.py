from typing import Optional, Literal

from pydantic import Field

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class EmbeddingsGenerationSettings(ExtraFieldsNotAllowedBaseModel):
    pooling_strategy: Literal['mean', 'last'] = Field(default='mean')
    item_embeddings_path: Optional[str] = Field(default=None)