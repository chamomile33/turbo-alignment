from pathlib import Path
from typing import Literal, List, Dict, Any, Optional, Union
from pydantic import Field

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.datasets.recommendation import RecommendationMultiDatasetSettings


class RecommendationMetricsExperimentSettings(ExtraFieldsNotAllowedBaseModel):
    model_settings: PreTrainedAdaptersModelSettings | PreTrainedModelSettings
    tokenizer_settings: TokenizerSettings
    
    dataset_settings: RecommendationMultiDatasetSettings = Field(...)
    item_embeddings_path: Path = Field(...)
    output_path: Path = Field(...)
    
    top_k: List[int] = Field(default=[10])
    batch_size: int = Field(default=128)
    max_tokens_count: int = Field(default=128)
    pooling_strategy: Literal['mean','last'] = Field(default='mean')
    
    use_accelerator: bool = Field(default=True)
    deepspeed_config: Optional[Union[Path, Dict[str, Any]]] = Field(default=None)
    fsdp_config: Optional[Dict[str, Any]] = Field(default=None) 