from typing import Any
import torch
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.registry import  CrossAttentionDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import DatasetSourceSettings, DatasetStrategy
from turbo_alignment.settings.datasets.recommendation import RecommendationDatasetSettings
from turbo_alignment.dataset.recommendation.models import UserRecommendationRecord
from turbo_alignment.dataset.recommendation.models import RecommendationDatasetRecord
from turbo_alignment.dataset.recommendation.recommendation import RecommendationDataset,UserRecommendationDataset
logger = get_project_logger()


@CrossAttentionDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class CrossAttentionUserRecommendationDataset(UserRecommendationDataset):
    
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
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
       
    
    def convert_records(self, records: list[UserRecommendationRecord]) -> list[dict[str, Any] | None]:
        result = []
        
        for record in records:
            events_tokens = self.tokenizer(
                record.events_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors=None,
            )
            
            client_id = record.meta['client_id']
            timestamp = record.meta['timestamp']
          
            encoded_record = {
                'id': record.id,
                'input_ids': torch.tensor(events_tokens['input_ids']),
                'attention_mask': torch.tensor(events_tokens['attention_mask']),
                'item_ids': record.item_ids,
                'meta': record.meta,
            }
            result.append(encoded_record)
            
        return result
    
    def get_slice(self, start: int, end: int):
        new_instance = self.__class__(
            source=self.source,
            settings=self.settings,
            tokenizer=self.tokenizer,
            read=False,
            seed=self.seed,
        )

        dataset_records = [self[idx] for idx in range(len(self))]

        new_instance.records = self.records[start:end]
        new_instance.original_records_map = {
            record['id']: self.get_original_record_by_id(record['id']) for record in dataset_records
        }

        return new_instance
    


@CrossAttentionDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class CrossAttentionRecommendationDataset(RecommendationDataset):
    
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
        
    
    def convert_records(self, records: list[RecommendationDatasetRecord]) -> list[dict[str, Any] | None]:
        result = []
        
        for record in records:
            events_tokens = self.tokenizer(
                record.events_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors=None,
            )
            
            item_ids = record.item_ids
            item_texts = record.items_text
            
            tokenized_items = []
            for item_id, item_text in zip(item_ids, item_texts):
                item_tokens = self.tokenizer(
                    item_text,
                    truncation=True,
                    max_length=self.settings.max_tokens_count // 4,
                    return_tensors=None,
                )
                
                tokenized_items.append({
                    'item_id': item_id,
                    'input_ids': torch.tensor(item_tokens['input_ids']),
                    'attention_mask': torch.tensor(item_tokens['attention_mask']),
                })
            
            client_id = record.meta['client_id']
            target_timestamp = record.meta['timestamp']
            
            encoded_record = {
                'record_id': record.id,
                'events_input_ids': torch.tensor(events_tokens['input_ids']),
                'events_attention_mask': torch.tensor(events_tokens['attention_mask']),
                'target_item_ids': item_ids,
                'tokenized_items': tokenized_items,
                'meta': record.meta,
            }
        
            result.append(encoded_record)
            
        return result