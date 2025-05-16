import math
from typing import Iterable

import numpy as np
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.dataset.recommendation import InferenceUserRecommendationDataset
from turbo_alignment.generators.embeddings import EmbeddingsGenerator
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.cherry_pick import RecommendationCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults
from turbo_alignment.dataset.recommendation.models import UserRecommendationRecord
from turbo_alignment.dataset.recommendation.collators import UserEmbeddingsCollator


class RecommendationCherryPickCallback(CherryPickCallbackBase[InferenceUserRecommendationDataset]):
    def __init__(
        self,
        cherry_pick_settings: RecommendationCherryPickSettings,
        datasets: Iterable[InferenceUserRecommendationDataset],
        metrics: list[Metric],
    ) -> None:
        super().__init__(cherry_pick_settings=cherry_pick_settings, datasets=datasets, metrics=metrics)
        self._custom_generation_settings = cherry_pick_settings.custom_generation_settings
        self._generator_transformers_settings = cherry_pick_settings.generator_transformers_settings

    def _get_dataset_metrics(
        self,
        dataset: InferenceUserRecommendationDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        accelerator: Accelerator = kwargs.get('accelerator', None)
        
        collator = UserEmbeddingsCollator(tokenizer=tokenizer)
        
        generator = EmbeddingsGenerator[UserRecommendationRecord](
            model=model,
            tokenizer=tokenizer,
            custom_generation_settings=self._custom_generation_settings,
        )
        
        if accelerator is not None:
            dataset = self._get_sharded_dataset(
                dataset=dataset,
                accelerator=accelerator,
            )

        generations = generator.generate_from_dataset(dataset)
        embeddings = [gen.embedding for gen in generations]
        true_items = [gen.item_ids for gen in generations]

        metric_outputs = [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset.source.name + '@@' + 'user_ids',
                        values=[gen.id for gen in generations],
                    )
                ]
            ),
        ]

        for metric in self._metrics:
            metric_results = metric.compute(
                model=model,
                dataset=dataset,
                embeddings=embeddings,
                true_items=true_items,
                dataset_name=dataset.source.name,
            )

            metric_outputs.extend(metric_results)
            
        return metric_outputs

    @staticmethod
    def _get_sharded_dataset(dataset: InferenceUserRecommendationDataset, accelerator: Accelerator) -> InferenceUserRecommendationDataset:
        rank_device = accelerator.process_index
        slice_size = math.ceil(len(dataset) / accelerator.num_processes)

        return dataset.get_slice(rank_device * slice_size, rank_device * slice_size + slice_size) 