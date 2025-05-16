from pathlib import Path
from typing import Any
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.registry import CollabAttentionDatasetTypeRegistry, CollaborativeAttentionEmbeddingsDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import DatasetSourceSettings, DatasetStrategy
from turbo_alignment.settings.datasets.recommendation import RecommendationDatasetSettings
from turbo_alignment.dataset.recommendation.models import CollabEmbedsDatasetRecord, CollabEmbedsUserRecommendationRecord, CollabEmbedsItemRecord
from turbo_alignment.dataset.recommendation.recommendation import (
    RecommendationDataset, 
    UserRecommendationDataset, 
    ItemDataset,
)

logger = get_project_logger()


@CollabAttentionDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class CollaborativeAttentionRecommendationDataset(RecommendationDataset):
    
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
    ) -> None:
        
        RecommendationDataset.__init__(
            self, 
            source=source, 
            settings=settings, 
            tokenizer=tokenizer, 
            seed=seed
        )
        
    def convert_records(self, records: list[CollabEmbedsDatasetRecord]) -> list[dict[str, Any] | None]:
      
        result = []
        
        for record in records:
            events_tokens = self.tokenizer(
                record.events_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors=None,
            )
            
            events_collaborative_keys = record.events_embeddings_keys 
            item_ids = record.item_ids
            item_texts = record.items_text
            items_embed_keys = record.items_embeddings_keys
            
            tokenized_items = []
            for i, (item_id, item_text, item_collaborative_keys) in enumerate(zip(item_ids, item_texts, items_embed_keys)):
                item_tokens = self.tokenizer(
                    item_text,
                    truncation=True,
                    max_length=self.settings.max_tokens_count // 4,
                    return_tensors=None,
                )
                
                tokenized_items.append({
                    'item_id': item_id,
                    'input_ids': item_tokens['input_ids'],
                    'attention_mask': item_tokens['attention_mask'],
                    'item_collaborative_keys': item_collaborative_keys,
                })
            
            result.append({
                'record_id': record.id,
                'events_input_ids': events_tokens['input_ids'],
                'events_attention_mask': events_tokens['attention_mask'],
                'events_collaborative_keys': events_collaborative_keys,
                'target_item_ids': item_ids,
                'tokenized_items': tokenized_items,
            })
            
        return result
    
    @staticmethod
    def _read_records(records):
        if isinstance(records, Path):
            return [CollabEmbedsDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [CollabEmbedsDatasetRecord(**record) for record in records]
        raise NotImplementedError



@CollabAttentionDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceCollaborativeAttentionUserRecommendationDataset(UserRecommendationDataset):

    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        collab_embed_token: str = "<collab-embed>",
        read: bool = True,
    ) -> None:
        
        UserRecommendationDataset.__init__(
            self,
            source=source, 
            settings=settings, 
            tokenizer=tokenizer, 
            seed=seed,
            read=read
        )
    
    
    def convert_records(self, records: list[CollabEmbedsUserRecommendationRecord]) -> list[dict[str, Any] | None]:
    
        result = []
        for record in records:
            events_tokens = self.tokenizer(
                record.events_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors=None,
            )
            
            collaborative_keys = record.events_embeddings_keys
            result.append({
                'id': record.id,
                'input_ids': events_tokens['input_ids'],
                'attention_mask': events_tokens['attention_mask'],
                'collaborative_keys': collaborative_keys,
                'item_ids': record.item_ids,
                'meta': record.meta
            })
            
        return result
    
    def get_slice(self, start: int, end: int):
        new_instance = self.__class__(
            source=self.source,
            settings=self.settings,
            tokenizer=self.tokenizer,
            read=False,
            seed=self.seed,
            collab_embed_token=self.collab_embed_token,
        )

        dataset_records = [self[idx] for idx in range(len(self))]

        new_instance.records = self.records[start:end]
        new_instance.original_records_map = {
            record['id']: self.get_original_record_by_id(record['id']) for record in dataset_records
        }

        return new_instance

    @staticmethod
    def _read_records(records):
        if isinstance(records, Path):
            return [CollabEmbedsUserRecommendationRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [CollabEmbedsUserRecommendationRecord(**record) for record in records]
        raise NotImplementedError


@CollaborativeAttentionEmbeddingsDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceCollaborativeAttentionItemDataset(ItemDataset):
   
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        collab_embed_token: str = "<collab-embed>",
        read: bool = True,
    ) -> None:

        ItemDataset.__init__(
            self,
            source=source, 
            settings=settings, 
            tokenizer=tokenizer, 
            seed=seed,
            read=read
        )
    
    
    def convert_records(self, records: list[CollabEmbedsItemRecord]) -> list[dict[str, Any] | None]:
        result = []
        
        for record in records:
            item_tokens = self.tokenizer(
                record.item_text,
                truncation=True,
                max_length=self.settings.max_tokens_count // 4,
                return_tensors=None,
            )
            
            collaborative_keys = record.item_embeddings_keys
            result.append({
                'id': record.id,
                'input_ids': item_tokens['input_ids'],
                'attention_mask': item_tokens['attention_mask'],
                'collaborative_keys': collaborative_keys,
                'item_id': record.item_id,
                'meta': record.meta
            })
            
        return result
    
    def get_slice(self, start: int, end: int):
        new_instance = self.__class__(
            source=self.source,
            settings=self.settings,
            tokenizer=self.tokenizer,
            read=False,
            seed=self.seed,
            collab_embed_token=self.collab_embed_token,
        )

        dataset_records = [self[idx] for idx in range(len(self))]

        new_instance.records = self.records[start:end]
        new_instance.original_records_map = {
            record['id']: self.get_original_record_by_id(record['id']) for record in dataset_records
        }

        return new_instance 

    @staticmethod
    def _read_records(records):
        if isinstance(records, Path):
            return [CollabEmbedsItemRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [CollabEmbedsItemRecord(**record) for record in records]
        raise NotImplementedError