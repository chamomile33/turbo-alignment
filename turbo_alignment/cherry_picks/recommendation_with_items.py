import math
from typing import Iterable, List, Optional
from pathlib import Path
from tqdm import tqdm 

import numpy as np
from accelerate import Accelerator
from accelerate.utils.operations import gather_object   
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.dataset.recommendation import InferenceUserRecommendationDataset, InferenceItemDataset
from turbo_alignment.dataset.recommendation.collab_embeds_dataset import InferenceCollaborativeUserRecommendationDataset
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.cherry_pick import RecommendationCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults
from turbo_alignment.dataset.recommendation.models import UserRecommendationRecord, ItemRecord, CollabEmbedsItemRecord, CollabEmbedsUserRecommendationRecord
from turbo_alignment.pipelines.inference.recommendation import UserEmbeddingsInferenceStrategy, ItemEmbeddingsInferenceStrategy
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.metrics.registry import MetricSettingsRegistry

from turbo_alignment.generators.embeddings import EmbeddingsGenerator
from turbo_alignment.dataset.recommendation.collab_embeds_dataset import InferenceCollaborativeItemDataset
from turbo_alignment.dataset.recommendation.collab_attention_dataset import InferenceCollaborativeAttentionItemDataset,InferenceCollaborativeAttentionUserRecommendationDataset
from turbo_alignment.dataset.recommendation.collab_cross_attention_dataset import CollabCrossAttentionUserRecommendationDataset

logger = get_project_logger()


class RecommendationWithItemsGenerationCherryPickCallback(CherryPickCallbackBase[InferenceUserRecommendationDataset]):
    def __init__(
        self,
        cherry_pick_settings: RecommendationCherryPickSettings,
        datasets: Iterable[InferenceUserRecommendationDataset],
        metrics: list[Metric],
        items_dataset: InferenceItemDataset,
        items_embeddings_output_path: Optional[str] = None,
    ) -> None:
        
        super().__init__(cherry_pick_settings=cherry_pick_settings, datasets=datasets, metrics=[])
        self._custom_generation_settings = cherry_pick_settings.custom_generation_settings
        self._generator_transformers_settings = cherry_pick_settings.generator_transformers_settings
        self._items_dataset = items_dataset
        self._items_embeddings_output_path = items_embeddings_output_path or "item_embeddings.jsonl"
       
        self._item_strategy = ItemEmbeddingsInferenceStrategy()
        self._user_strategy = UserEmbeddingsInferenceStrategy()
        self._metric_settings = cherry_pick_settings.metric_settings

    def _get_dataset_metrics(
        self,
        dataset: InferenceUserRecommendationDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        accelerator: Accelerator = kwargs.get('accelerator', None)
    
        item_ids, item_embeddings = self._generate_item_embeddings(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator
        )
        
        user_embeddings, user_ids, true_items = self._generate_user_embeddings(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator
        )

        metric_outputs = [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset.source.name + '@@' + 'user_ids',
                        values=user_ids,
                    )
                ]
            ),
        ]
        metrics = self._create_metrics(item_generation_output=(item_ids, item_embeddings))
        
        for metric in metrics:
            metric_results = metric.compute(
                model=model,
                dataset=dataset,
                embeddings=user_embeddings,
                true_items=true_items,
                dataset_name=dataset.source.name,
            )

            metric_outputs.extend(metric_results)

        return metric_outputs
    
    def _create_metrics(self, item_generation_output, item_embeddings_path: str = '') -> List[Metric]:
        
        metrics = []
        for metric_setting in self._metric_settings:

            parameters = dict(metric_setting.parameters)
            parameters['item_embeddings_path'] = item_embeddings_path
            
            metric = Metric.by_name(metric_setting.type)(
                MetricSettingsRegistry.by_name(metric_setting.type)(**parameters),
                  item_generation_output = item_generation_output
            )
            metrics.append(metric)
        
        return metrics
    

    def _generate_item_embeddings(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Optional[Accelerator] = None,
    ) -> str:
        logger.info("Начало генерации эмбеддингов айтемов")
        
        items_dataset = self._items_dataset
        embeddings_settings = self._custom_generation_settings
    
        if isinstance(self._items_dataset, InferenceCollaborativeItemDataset) or isinstance(self._items_dataset, InferenceCollaborativeAttentionItemDataset):
            item_cls = CollabEmbedsItemRecord
        else:
            item_cls = ItemRecord

        generator = EmbeddingsGenerator[item_cls](
                model=model,
                tokenizer=tokenizer,
                custom_generation_settings=embeddings_settings,
                batch=self._metric_settings[0].parameters['batch_size'] * 4,
                accelerator=accelerator
        )

        item_generations = generator.generate_from_dataset(items_dataset)
        
        output_path = Path(self._items_embeddings_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if accelerator is not None:
            item_generations = gather_object(item_generations)
            unique_items = set()
            item_ids = []
            item_embeddings = []
          
            for item in tqdm(item_generations):
                item_id = item.item_id
                item_text = item.item_text
                key = (item_id, item_text)
                if key not in unique_items:
                    unique_items.add(key)
                    item_ids.append(item_id)
                    item_embeddings.append(np.array(item.embedding))


        print(f'Всего {len(item_ids)} эмбеддингов айтемов')
        return item_ids, item_embeddings

    def _generate_user_embeddings(
        self,
        dataset: InferenceUserRecommendationDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Optional[Accelerator] = None,
    ) -> tuple[List[np.ndarray], List[str], List[List[str]]]:
    
        logger.info("Начало генерации эмбеддингов пользователей")
        embeddings_settings = self._custom_generation_settings
        
        if (
            isinstance(dataset, InferenceCollaborativeUserRecommendationDataset) or 
            isinstance(dataset, InferenceCollaborativeAttentionUserRecommendationDataset) or
            isinstance(dataset, CollabCrossAttentionUserRecommendationDataset)
        ):
            user_cls = CollabEmbedsUserRecommendationRecord
        else:
            user_cls = UserRecommendationRecord

        generator = EmbeddingsGenerator[user_cls](
                model=model,
                tokenizer=tokenizer,
                custom_generation_settings=embeddings_settings,
                batch=self._metric_settings[0].parameters['batch_size'],
                accelerator=accelerator
        )

        user_generations = generator.generate_from_dataset(dataset)
        user_embeddings = [gen.embedding for gen in user_generations]
        user_ids = [gen.id for gen in user_generations]
        true_items = [gen.item_ids for gen in user_generations]
        
        logger.info(f"Сгенерировано {len(user_embeddings)} эмбеддингов пользователей")
        
        return user_embeddings, user_ids, true_items

    @staticmethod
    def _get_sharded_dataset(dataset: InferenceUserRecommendationDataset, accelerator: Accelerator) -> InferenceUserRecommendationDataset:
        rank_device = accelerator.process_index
        slice_size = math.ceil(len(dataset) / accelerator.num_processes)

        return dataset.get_slice(rank_device * slice_size, rank_device * slice_size + slice_size)
    
    @staticmethod
    def _get_sharded_items_dataset(dataset: InferenceItemDataset, accelerator: Accelerator) -> InferenceItemDataset:
        rank_device = accelerator.process_index
        slice_size = math.ceil(len(dataset) / accelerator.num_processes)

        return dataset.get_slice(rank_device * slice_size, rank_device * slice_size + slice_size)
    