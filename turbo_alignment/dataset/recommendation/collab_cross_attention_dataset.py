from pathlib import Path
from typing import Any
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.registry import CollabCrossAttentionDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import DatasetSourceSettings, DatasetStrategy
from turbo_alignment.settings.datasets.recommendation import RecommendationDatasetSettings
from turbo_alignment.dataset.recommendation.models import CollabEmbedsDatasetRecord, CollabEmbedsUserRecommendationRecord
from pathlib import Path

from turbo_alignment.dataset.recommendation.collab_attention_dataset import CollaborativeAttentionRecommendationDataset, InferenceCollaborativeAttentionUserRecommendationDataset

logger = get_project_logger()

@CollabCrossAttentionDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class CollabCrossAttentionRecommendationDataset(CollaborativeAttentionRecommendationDataset):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
    ) -> None:
        
        CollaborativeAttentionRecommendationDataset.__init__(
            self,
            source=source, 
            settings=settings, 
            tokenizer=tokenizer, 
            seed=seed,
            read=read
        )
    
    def convert_records(self, records: list[CollabEmbedsDatasetRecord]) -> list[dict[str, Any] | None]:
        result = super().convert_records(records)
        for i in range(len(records)):
            record = records[i]
            assert record.id == result[i]['record_id']
            result[i]['meta'] = record.meta
        return result 
    

    @staticmethod
    def _read_records(records):
        if isinstance(records, Path):
            return [CollabEmbedsDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [CollabEmbedsDatasetRecord(**record) for record in records]
        raise NotImplementedError



@CollabCrossAttentionDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class CollabCrossAttentionUserRecommendationDataset(InferenceCollaborativeAttentionUserRecommendationDataset):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
    ) -> None:

        InferenceCollaborativeAttentionUserRecommendationDataset.__init__(
            self,
            source=source, 
            settings=settings, 
            tokenizer=tokenizer, 
            seed=seed,
            read=read
        )
    
    def convert_records(self, records: list[CollabEmbedsUserRecommendationRecord]) -> list[dict[str, Any] | None]:
        result = super().convert_records(records)
        for i in range(len(records)):
            record = records[i]
            assert record.id == result[i]['id']
            result[i]['meta'] = record.meta
        return result
        
    @staticmethod
    def _read_records(records):
        if isinstance(records, Path):
            return [CollabEmbedsUserRecommendationRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [CollabEmbedsUserRecommendationRecord(**record) for record in records]
        raise NotImplementedError