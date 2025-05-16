from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.generators.embeddings import EmbeddingsGenerationSettings
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
    SingleModelInferenceSettings,
)
from pydantic import Field


class EmbeddingsGenerationModelSettings(ExtraFieldsNotAllowedBaseModel):
    custom_settings: EmbeddingsGenerationSettings


class RecommendationModelInferenceSettings(SingleModelInferenceSettings):
    generation_settings: List[EmbeddingsGenerationModelSettings]


class RecommendationInferenceExperimentSettings(InferenceExperimentSettings):
    inference_settings: List[RecommendationModelInferenceSettings]
    
    use_accelerator: bool = Field(default=True)
    deepspeed_config: Optional[Union[Path, Dict[str, Any]]] = Field(default=None)
    fsdp_config: Optional[Dict[str, Any]] = Field(default=None) 