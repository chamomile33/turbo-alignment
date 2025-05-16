from typing import Callable, Optional, Union

from torch.utils.data import Dataset, ConcatDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.dataset.recommendation import RecommendationDataCollator, RecommendationDataset, TimeAwareRecommendationDataset
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.pipelines.train.recommendation import RecommendationTrainExperimentSettings
from turbo_alignment.trainers.recommendation import RecommendationTrainer
from turbo_alignment.trainers.recommendation_loss import RecommendationLoss
from turbo_alignment.dataset.recommendation import (
    InferenceUserRecommendationDataset,
    InferenceItemDataset,
    InferenceTimeAwareUserRecommendationDataset,
    InferenceCollaborativeUserRecommendationDataset,
    InferenceCollaborativeItemDataset,
    CollaborativeRecommendationDataset
)
from turbo_alignment.dataset.recommendation.cross_attention_dataset import CrossAttentionUserRecommendationDataset, CrossAttentionRecommendationDataset
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.cherry_picks.recommendation import RecommendationCherryPickCallback
from turbo_alignment.cherry_picks.recommendation_with_items import RecommendationWithItemsGenerationCherryPickCallback
from turbo_alignment.settings.cherry_pick import RecommendationWithItemsCherryPickSettings
from turbo_alignment.dataset.recommendation.collators import TimeAwareRecommendationDataCollator
from turbo_alignment.dataset.recommendation.collab_embeds_collator import CollaborativeRecommendationDataCollator
from turbo_alignment.dataset.recommendation.cross_attention_collator import CrossAttentionRecommendationDataCollator
from turbo_alignment.dataset.recommendation.collab_attention_collator import CollaborativeAttentionRecommendationDataCollator
from turbo_alignment.dataset.recommendation.collab_attention_dataset import InferenceCollaborativeAttentionUserRecommendationDataset, InferenceCollaborativeAttentionItemDataset, CollaborativeAttentionRecommendationDataset
from turbo_alignment.dataset.recommendation.collab_cross_attention_collator import CollabCrossAttentionRecommendationDataCollator
from turbo_alignment.dataset.recommendation.collab_cross_attention_dataset import CollabCrossAttentionUserRecommendationDataset, CollabCrossAttentionRecommendationDataset

logger = get_project_logger()


class TrainRecommendationStrategy(BaseTrainStrategy[RecommendationTrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: RecommendationTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Callable:
        if experiment_settings.train_dataset_settings.dataset_type == 'collab_cross_attention':
            return CollabCrossAttentionRecommendationDataCollator(tokenizer=tokenizer)
        elif experiment_settings.train_dataset_settings.dataset_type == 'collab_attention':
            return CollaborativeAttentionRecommendationDataCollator(tokenizer=tokenizer)
        elif experiment_settings.train_dataset_settings.dataset_type == 'time_aware_recommendation':
            return TimeAwareRecommendationDataCollator(tokenizer=tokenizer)
        elif experiment_settings.train_dataset_settings.dataset_type == 'collab_embeds':
            return CollaborativeRecommendationDataCollator(tokenizer=tokenizer)
        elif experiment_settings.train_dataset_settings.dataset_type == 'cross_attention':
            return CrossAttentionRecommendationDataCollator(tokenizer=tokenizer)
        else:
            return RecommendationDataCollator(tokenizer=tokenizer)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: RecommendationTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Optional[Union[RecommendationCherryPickCallback, RecommendationWithItemsGenerationCherryPickCallback]]:
     
        cherry_pick_settings = experiment_settings.cherry_pick_settings
        if cherry_pick_settings is None:
            return None
       
        if experiment_settings.train_dataset_settings.dataset_type == 'collab_cross_attention':
            dataset_cls = CollabCrossAttentionUserRecommendationDataset
        elif experiment_settings.train_dataset_settings.dataset_type == 'time_aware_recommendation':
            dataset_cls = InferenceTimeAwareUserRecommendationDataset
        elif experiment_settings.train_dataset_settings.dataset_type == 'collab_attention':
            dataset_cls = InferenceCollaborativeAttentionUserRecommendationDataset
        elif experiment_settings.train_dataset_settings.dataset_type == 'collab_embeds':
            dataset_cls = InferenceCollaborativeUserRecommendationDataset
        elif experiment_settings.train_dataset_settings.dataset_type == 'cross_attention':
            dataset_cls = CrossAttentionUserRecommendationDataset
        else:
            dataset_cls = InferenceUserRecommendationDataset

        
        user_datasets = DatasetLoader[dataset_cls](dataset_cls).load_datasets(
            cherry_pick_settings.dataset_settings,
            tokenizer=tokenizer,
            strategy=DatasetStrategy.INFERENCE,
            seed=experiment_settings.seed,
        )
        if isinstance(cherry_pick_settings, RecommendationWithItemsCherryPickSettings):
            if cherry_pick_settings.items_dataset_settings.dataset_type == 'collab_embeds_embeddings':
                item_dataset_cls = InferenceCollaborativeItemDataset
            elif cherry_pick_settings.items_dataset_settings.dataset_type == 'collab_attention_embeddings':
                item_dataset_cls = InferenceCollaborativeAttentionItemDataset
            else:
                item_dataset_cls = InferenceItemDataset

            item_datasets = DatasetLoader[item_dataset_cls](item_dataset_cls).load_datasets(
                cherry_pick_settings.items_dataset_settings,
                tokenizer=tokenizer,
                strategy=DatasetStrategy.INFERENCE,
                seed=experiment_settings.seed,
            )
            
            if not item_datasets:
                raise ValueError("Не найдены датасеты айтемов для RecommendationWithItemsGenerationCherryPickCallback")
                
            return RecommendationWithItemsGenerationCherryPickCallback(
                cherry_pick_settings=cherry_pick_settings,
                datasets=user_datasets,
                metrics=[],  
                items_dataset=item_datasets[0],
                items_embeddings_output_path=cherry_pick_settings.items_embeddings_output_path,
            )
        else:
            metrics = [
                Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
                for metric in cherry_pick_settings.metric_settings
            ]
            
            return RecommendationCherryPickCallback(
                cherry_pick_settings=cherry_pick_settings,
                datasets=user_datasets,
                metrics=metrics,
            )

    @staticmethod
    def _get_training_args(experiment_settings: RecommendationTrainExperimentSettings) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_trainer(
        training_args: TrainingArguments,
        experiment_settings: RecommendationTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: DataCollatorMixin,
        **_kwargs,
    ) -> RecommendationTrainer:

        model.config.use_cache = not experiment_settings.trainer_settings.gradient_checkpointing

        recommendation_loss = RecommendationLoss(
            pooling_strategy=experiment_settings.loss_settings.pooling_strategy,
            temperature=experiment_settings.loss_settings.temperature,
            gather_items_in_batch=experiment_settings.loss_settings.gather_items_in_batch
        )

        return RecommendationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            recommendation_loss=recommendation_loss,
        )

    def _load_datasets(self) -> tuple[Dataset, Dataset]:
        logger.info('Loading train dataset')
        if self.experiment_settings.train_dataset_settings.dataset_type == 'collab_cross_attention':
            dataset_cls = CollabCrossAttentionRecommendationDataset
        elif self.experiment_settings.train_dataset_settings.dataset_type == 'time_aware_recommendation':
            dataset_cls = TimeAwareRecommendationDataset
        elif self.experiment_settings.train_dataset_settings.dataset_type == 'collab_embeds':
            dataset_cls = CollaborativeRecommendationDataset
        elif self.experiment_settings.train_dataset_settings.dataset_type == 'cross_attention':
            dataset_cls = CrossAttentionRecommendationDataset   
        elif self.experiment_settings.train_dataset_settings.dataset_type == 'collab_attention':
            dataset_cls = CollaborativeAttentionRecommendationDataset
        else:
            dataset_cls = RecommendationDataset

        train_dataset = ConcatDataset(
            datasets = DatasetLoader[dataset_cls](dataset_cls).load_datasets(
            self.experiment_settings.train_dataset_settings,
            tokenizer=self.tokenizer,
            strategy=DatasetStrategy.TRAIN,
            seed=self.experiment_settings.seed,
        ))

        logger.info('Loading val dataset')
        val_dataset = ConcatDataset(
            datasets = DatasetLoader[dataset_cls](dataset_cls).load_datasets(
            self.experiment_settings.val_dataset_settings,
            tokenizer=self.tokenizer,
            strategy=DatasetStrategy.TRAIN,
            seed=self.experiment_settings.seed,
        ))

        return train_dataset, val_dataset

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample example:\n{dataset[0]}')
        
        batch = collator([dataset[0], dataset[1]])
        logger.info(f'Events input ids: {batch["events_input_ids"].shape}')
        logger.info(f'Events attention mask: {batch["events_attention_mask"].shape}')
        logger.info(f'Item input ids: {batch["item_input_ids"].shape}')
        logger.info(f'Item attention mask: {batch["item_attention_mask"].shape}') 