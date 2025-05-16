from pydantic import Field

from turbo_alignment.settings.pipelines.train.base import (
    BaseTrainExperimentSettings,
    TrainerSettings,
)
from turbo_alignment.settings.datasets.recommendation import (
    RecommendationMultiDatasetSettings,
)
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.cherry_pick import RecommendationWithItemsCherryPickSettings


class RecommendationLossSettings(ExtraFieldsNotAllowedBaseModel):

    pooling_strategy: str = Field(
        default="mean",
    )
    temperature: float = Field(
        default=1,
    )
    gather_items_in_batch: bool = Field(
        default=True
    )


class RecommendationTrainExperimentSettings(BaseTrainExperimentSettings):
    trainer_settings: TrainerSettings
    train_dataset_settings: RecommendationMultiDatasetSettings
    val_dataset_settings: RecommendationMultiDatasetSettings
    loss_settings: RecommendationLossSettings = Field(
        default_factory=RecommendationLossSettings,
    )
    cherry_pick_settings: RecommendationWithItemsCherryPickSettings | None = None