from typing import Optional

from turbo_alignment.settings.base import  ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.base import MultiDatasetSettings
from turbo_alignment.settings.datasets.chat import ChatMultiDatasetSettings
from turbo_alignment.settings.datasets.recommendation import RecommendationMultiDatasetSettings
from turbo_alignment.settings.datasets.classification import (
    ClassificationMultiDatasetSettings,
)
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceMultiDatasetSettings,
)
from turbo_alignment.settings.generators.embeddings import EmbeddingsGenerationSettings
from turbo_alignment.settings.metric import MetricSettings
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings


class CherryPickSettings(ExtraFieldsNotAllowedBaseModel):
    dataset_settings: MultiDatasetSettings
    metric_settings: list[MetricSettings]

class RMCherryPickSettings(CherryPickSettings):
    dataset_settings: PairPreferenceMultiDatasetSettings


class ClassificationCherryPickSettings(CherryPickSettings):
    dataset_settings: ClassificationMultiDatasetSettings


class GenerationSettings(CherryPickSettings):
    generator_transformers_settings: GeneratorTransformersSettings
    custom_generation_settings: ExtraFieldsNotAllowedBaseModel


class ChatCherryPickSettings(GenerationSettings):
    dataset_settings: ChatMultiDatasetSettings


class RecommendationCherryPickSettings(CherryPickSettings):
    custom_generation_settings: EmbeddingsGenerationSettings
    generator_transformers_settings: Optional[GeneratorTransformersSettings] = None

class RecommendationWithItemsCherryPickSettings(RecommendationCherryPickSettings):
    items_dataset_settings: RecommendationMultiDatasetSettings
    items_embeddings_output_path: Optional[str] = None
    dataset_settings: RecommendationMultiDatasetSettings
