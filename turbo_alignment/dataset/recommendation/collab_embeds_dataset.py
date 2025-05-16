from pathlib import Path
from typing import Any, List, Dict
from typing_extensions import Self
import torch
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.registry import CollabEmbedsEmbeddingsDatasetTypeRegistry, CollabEmbedsDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import DatasetSourceSettings, DatasetStrategy
from turbo_alignment.settings.datasets.recommendation import RecommendationDatasetSettings
from turbo_alignment.dataset.recommendation.models import CollabEmbedsDatasetRecord, CollabEmbedsItemRecord, CollabEmbedsUserRecommendationRecord
from turbo_alignment.dataset.recommendation.recommendation import RecommendationDataset,UserRecommendationDataset, ItemDataset
from pathlib import Path
from turbo_alignment.dataset.recommendation.collab_attention_collator import GLOBAL_EMBEDDINGS_DICTS

logger = get_project_logger()

class CollaborativeEmbeddingsMixin:

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase,
        collab_embed_token: str = "<collab-embed>",
        *args, 
        **kwargs
    ):
        self.collab_embed_token = collab_embed_token
        self.tokenizer = tokenizer
        self.embeddings_dicts = GLOBAL_EMBEDDINGS_DICTS
        if self.collab_embed_token not in tokenizer.get_vocab():
            logger.warning(f"Токен {self.collab_embed_token} не найден в словаре токенизатора.")
    
    def get_collaborative_embeddings(self, embed_keys: List[Dict[str, str]]) -> torch.Tensor: 
        embeds = []
        for key_info in embed_keys:
            embed_type = key_info.get('type')
            embed_key = key_info.get('key')
            
            if not embed_type or not embed_key:
                logger.warning(f"Пропущен ключ без type или key: {key_info}")
                continue
                
            if embed_type not in self.embeddings_dicts:
                raise Exception(f"Тип эмбеддинга {embed_type} не найден в embeddings_dicts")
                
            if embed_key not in self.embeddings_dicts[embed_type]:
                raise Exception(f"Ключ {embed_key} не найден в словаре эмбеддингов типа {embed_type}")
                
            embeds.append(torch.tensor(self.embeddings_dicts[embed_type][embed_key]))
            
        if not embeds:
            return torch.tensor([])
        return torch.stack(embeds)
    
    def get_collab_embed_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.collab_embed_token)
    
@CollabEmbedsDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class CollaborativeRecommendationDataset(RecommendationDataset, CollaborativeEmbeddingsMixin):
    
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
        collab_embed_token: str = "<collab-embed>",
    ) -> None:
        
        CollaborativeEmbeddingsMixin.__init__(
            self,
            tokenizer=tokenizer,
            collab_embed_token=collab_embed_token
        )
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
            
            events_embed_keys = record.events_embeddings_keys
            events_embeds = self.get_collaborative_embeddings(events_embed_keys)
            
            item_ids = record.item_ids
            item_texts = record.items_text
            items_embed_keys = record.items_embeddings_keys
            
            tokenized_items = []
            for item_id, item_text, item_embed_keys in zip(item_ids, item_texts, items_embed_keys):
                item_tokens = self.tokenizer(
                    item_text,
                    truncation=True,
                    max_length=self.settings.max_tokens_count // 4,
                    return_tensors=None,
                )
                item_embeds = self.get_collaborative_embeddings(item_embed_keys)
                
                tokenized_items.append({
                    'item_id': item_id,
                    'input_ids': torch.tensor(item_tokens['input_ids']),
                    'attention_mask': torch.tensor(item_tokens['attention_mask']),
                    'collab_embeddings': item_embeds,
                })
            
            result.append({
                'record_id': record.id,
                'events_input_ids': torch.tensor(events_tokens['input_ids']),
                'events_attention_mask': torch.tensor(events_tokens['attention_mask']),
                'events_collab_embeddings': events_embeds,
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


@CollabEmbedsDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceCollaborativeUserRecommendationDataset(UserRecommendationDataset, CollaborativeEmbeddingsMixin):
    
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
        collab_embed_token: str = "<collab-embed>",
    ) -> None:
        
        CollaborativeEmbeddingsMixin.__init__(
            self,
            tokenizer=tokenizer,
            collab_embed_token=collab_embed_token
        )
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
            input_ids = self.tokenizer(
                record.events_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors='np',
            )['input_ids'].squeeze(0)
            
            input_ids_tensor = torch.tensor(input_ids)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)
            events_embed_keys = record.events_embeddings_keys
            events_embeds = self.get_collaborative_embeddings(events_embed_keys)
            
            result.append({
                'id': record.id,
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask_tensor,
                'collab_embeddings': events_embeds,
                'item_ids': record.item_ids,
                'meta': record.meta
            })
            
        return result
    
    def get_slice(self, start: int, end: int) -> Self:
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
    

@CollabEmbedsEmbeddingsDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceCollaborativeItemDataset(ItemDataset, CollaborativeEmbeddingsMixin):
    
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
        collab_embed_token: str = "<collab-embed>",
    ) -> None:
    
        CollaborativeEmbeddingsMixin.__init__(
            self,
            tokenizer=tokenizer,
            collab_embed_token=collab_embed_token
        )
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
            input_ids = self.tokenizer(
                record.item_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors='np',
            )['input_ids'].squeeze(0)
            
            input_ids_tensor = torch.tensor(input_ids)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)
            item_embed_keys = record.item_embeddings_keys
            item_embeds = self.get_collaborative_embeddings(item_embed_keys)
            
            result.append({
                'id': record.id,
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask_tensor,
                'collab_embeddings': item_embeds,
                'item_id': record.item_id,
                'meta': record.meta
            })
            
        return result
    
    def get_slice(self, start: int, end: int) -> Self:
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