from typing import Optional, Literal

from pydantic import Field

from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)


class RecommendationDatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.RECOMMENDATION, DatasetType.EMBEDDINGS, DatasetType.TIME_AWARE_RECOMMENDATION,
                           DatasetType.COLLAB_EMBEDS_EMBEDDINGS, DatasetType.COLLAB_EMBEDS, DatasetType.CROSS_ATTENTION,
                           DatasetType.COLLAB_ATTENTION, DatasetType.COLLAB_ATTENTION_EMBEDDINGS, DatasetType.COLLAB_CROSS_ATTENTION] = DatasetType.RECOMMENDATION
    max_tokens_count: int = Field(default=512)
    ignore_errors: bool = Field(default=False)
    top_k: int = Field(default=100)
    item_embeddings_path: Optional[str] = Field(default=None)


class RecommendationMultiDatasetSettings(RecommendationDatasetSettings, MultiDatasetSettings):
    ...