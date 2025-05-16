from pathlib import Path
from typing import Any, overload
from typing_extensions import Self
import torch
from transformers import PreTrainedTokenizerBase
from datetime import datetime

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.registry import EmbeddingsDatasetTypeRegistry,RecommendationDatasetTypeRegistry,TimeAwareRecommendationDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import DatasetSourceSettings, DatasetStrategy
from turbo_alignment.settings.datasets.recommendation import RecommendationDatasetSettings
from turbo_alignment.dataset.recommendation.models import UserRecommendationRecord, ItemRecord, TimeAwareRecommendationDatasetRecord
from turbo_alignment.dataset.recommendation.models import RecommendationDatasetRecord, TimeAwareUserRecommendationRecord
logger = get_project_logger()

@RecommendationDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class RecommendationDataset(AlignmentDataset[RecommendationDatasetRecord]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
    ) -> None:
        super().__init__(source=source, settings=settings, tokenizer=tokenizer, seed=seed)
        self._read()
        self.settings: RecommendationDatasetSettings = settings

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
            
            result.append({
                'record_id': record.id,
                'events_input_ids': torch.tensor(events_tokens['input_ids']),
                'events_attention_mask': torch.tensor(events_tokens['attention_mask']),
                'target_item_ids': item_ids, 
                'tokenized_items': tokenized_items, 
            })    
        return result

    
    @staticmethod
    @overload
    def _read_records(records: Path) -> list[UserRecommendationRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[UserRecommendationRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[UserRecommendationRecord]:
        if isinstance(records, Path):
            return [RecommendationDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [RecommendationDatasetRecord(**record) for record in records]
        raise NotImplementedError


         
class UserRecommendationDataset(AlignmentDataset[UserRecommendationRecord]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
    ) -> None:
        super().__init__(source=source, settings=settings, tokenizer=tokenizer, seed=seed)
        self.settings: RecommendationDatasetSettings = settings
        if read:
            self._read()

    def _encode(
        self,
        records: list[UserRecommendationRecord]
    ) -> list[dict[str, Any] | None]:
        logger.info(f'Tokenizing user dataset {self.source.name}')
        
        output: list[dict[str, Any] | None] = []
        
        for record in records:
            input_ids = self.tokenizer(
                record.events_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors='np',
            )['input_ids'].squeeze(0)
            
            encoded_record: dict[str, Any] = {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.ones(input_ids.shape),
            }
        
            encoded_record.update({
                'id': record.id,
                'item_ids': record.item_ids,
                'meta': record.meta
            })
            
            output.append(encoded_record)
                
        return output

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[UserRecommendationRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[UserRecommendationRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[UserRecommendationRecord]:
        if isinstance(records, Path):
            return [UserRecommendationRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [UserRecommendationRecord(**record) for record in records]
        raise NotImplementedError


class ItemDataset(AlignmentDataset[ItemRecord]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True,
    ) -> None:
        super().__init__(source=source, settings=settings, tokenizer=tokenizer, seed=seed)
        self.settings: RecommendationDatasetSettings = settings
        if read:
            self._read()

    def _encode(
        self,
        records: list[ItemRecord]
    ) -> list[dict[str, Any] | None]:
        logger.info(f'Tokenizing item dataset {self.source.name}')
        
        output: list[dict[str, Any] | None] = []
        
        for record in records:
            input_ids = self.tokenizer(
                record.item_text,
                truncation=True,
                max_length=self.settings.max_tokens_count,
                return_tensors='np',
            )['input_ids'].squeeze(0)
            

            encoded_record: dict[str, Any] = {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.ones(input_ids.shape),
            }
        
            encoded_record.update({
                'id': record.id,
                'item_id': record.item_id,
                'meta': record.meta
            })
            
            output.append(encoded_record)
                
        return output
        

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[ItemRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[ItemRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[ItemRecord]:
        if isinstance(records, Path):
            return [ItemRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [ItemRecord(**record) for record in records]
        raise NotImplementedError


@RecommendationDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceUserRecommendationDataset(UserRecommendationDataset):
    def convert_records(self, records: list[UserRecommendationRecord]) -> list[dict[str, Any] | None]:
        return self._encode(records)
    
    def get_slice(self, start: int, end: int) -> Self:
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


@EmbeddingsDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceItemDataset(ItemDataset):
    def convert_records(self, records: list[ItemRecord]) -> list[dict[str, Any] | None]:
        return self._encode(records)
    
    def get_slice(self, start: int, end: int) -> Self:
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


class TimeAwareMixin:
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        self.end_of_event_token = "<end-of-event>"
    
        if self.end_of_event_token not in tokenizer.get_vocab():
            logger.warning(f"Токен {self.end_of_event_token} не найден в словаре токенизатора.")
    
    def _timestamp_to_time_vector(self, timestamp_ns: int) -> list[float]:
        if timestamp_ns == 0:
            return [-1, -1, -1, -1, -1, -1]
        dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000)

        def week_of_month(dt):
            first_day = dt.replace(day=1)
            dom = dt.day
            adjusted_dom = dom + first_day.weekday()
            return (adjusted_dom - 1) // 7 + 1

        year = dt.year / 3000.0
        month = dt.month / 12.0
        week = week_of_month(dt) / 5.0
        day = dt.day / 31.0
        weekday = dt.weekday() / 6.0
        hour = dt.hour / 24.0

        return [year, month, week, day, weekday, hour]
    
    def _create_time_vectors(self, input_ids: list, timestamps: list[int]) -> torch.Tensor:
        end_token_id = self.tokenizer.convert_tokens_to_ids(self.end_of_event_token)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int32)
        end_token_mask = (input_ids_tensor == end_token_id)
        time_vectors_raw = torch.tensor([self._timestamp_to_time_vector(ts) for ts in timestamps], 
                                       dtype=torch.float32)
        
        seq_len = len(input_ids)
        segment_ids = torch.zeros(seq_len, dtype=torch.long)
        end_token_indices = end_token_mask.nonzero().squeeze(-1)
        segment_ids = torch.cumsum(end_token_mask, dim=0) 
        segment_ids[end_token_indices] -= 1
        
        time_vectors = time_vectors_raw[segment_ids]
        
        return time_vectors
    
    def process_time_data(self, input_ids: list, timestamps: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end_token_id = self.tokenizer.convert_tokens_to_ids(self.end_of_event_token)
        time_vectors = self._create_time_vectors(input_ids, timestamps)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int32)
        end_token_mask = (input_ids_tensor != end_token_id)
        
        filtered_input_ids = input_ids_tensor[end_token_mask]
        attention_mask = torch.ones_like(input_ids_tensor)
        filtered_attention_mask = attention_mask[end_token_mask]
        filtered_time_vectors = time_vectors[end_token_mask]
        
        return filtered_input_ids, filtered_attention_mask, filtered_time_vectors


@TimeAwareRecommendationDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class TimeAwareRecommendationDataset(RecommendationDataset, TimeAwareMixin):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True
    ) -> None:
    
        TimeAwareMixin.__init__(
            self,
            tokenizer=tokenizer 
        )
        RecommendationDataset.__init__(
            self, 
            source=source, 
            settings=settings, 
            tokenizer=tokenizer, 
            seed=seed
        )    
    
    def convert_records(self, records: list[TimeAwareRecommendationDatasetRecord]) -> list[dict[str, Any] | None]:
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
            timestamps = record.events_timestamps
            
            filtered_input_ids, filtered_attention_mask, filtered_time_vectors = self.process_time_data(
                events_tokens['input_ids'], 
                timestamps
            )
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
            
            result.append({
                'record_id': record.id,
                'events_input_ids': filtered_input_ids,
                'events_attention_mask': filtered_attention_mask,
                'time_vectors': filtered_time_vectors,
                'target_item_ids': item_ids,
                'tokenized_items': tokenized_items,
            })
            
        return result
    
    @staticmethod
    def _read_records(records) -> list[ItemRecord]:
        if isinstance(records, Path):
            return [TimeAwareRecommendationDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [TimeAwareRecommendationDatasetRecord(**record) for record in records]
        raise NotImplementedError

@TimeAwareRecommendationDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceTimeAwareUserRecommendationDataset(UserRecommendationDataset, TimeAwareMixin):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: RecommendationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        read: bool = True
    ) -> None:
      
        TimeAwareMixin.__init__(
            self,
            tokenizer=tokenizer
        )
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
            timestamps = record.events_timestamps
            filtered_input_ids, filtered_attention_mask, filtered_time_vectors = self.process_time_data(
                events_tokens['input_ids'], 
                timestamps
            )
            result.append({
                'id': record.id,
                'input_ids': filtered_input_ids,
                'attention_mask': filtered_attention_mask,
                'time_vectors': filtered_time_vectors,
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
            end_of_event_token=self.end_of_event_token,
        )

        dataset_records = [self[idx] for idx in range(len(self))]

        new_instance.records = self.records[start:end]
        new_instance.original_records_map = {
            record['id']: self.get_original_record_by_id(record['id']) for record in dataset_records
        }

        return new_instance

    @staticmethod
    def _read_records(records) -> list[TimeAwareUserRecommendationRecord]:
        if isinstance(records, Path):
            return [TimeAwareUserRecommendationRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [TimeAwareUserRecommendationRecord(**record) for record in records]
        raise NotImplementedError