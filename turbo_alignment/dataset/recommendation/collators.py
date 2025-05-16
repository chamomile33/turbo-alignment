from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class RecommendationDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            'events_input_ids': [],
            'events_attention_mask': [],
            'item_input_ids': [],
            'item_attention_mask': [],
            'item_ids': [],
            'target_item_ids': [],
        }
        
        for example in examples:

            tokenized_items = example['tokenized_items']
            batch['events_input_ids'].append(example['events_input_ids'])
            batch['events_attention_mask'].append(example['events_attention_mask'])
            
            target_ids = [item['item_id'] for item in tokenized_items]
            batch['target_item_ids'].append(target_ids)
            for item in tokenized_items:
                batch['item_input_ids'].append(item['input_ids'])
                batch['item_attention_mask'].append(item['attention_mask'])
                batch['item_ids'].append(item['item_id'])
            
        batch['item_ids'] = torch.tensor(batch['item_ids'])
        batch_size = len(batch['events_input_ids'])
        max_events_length = max(len(seq) for seq in batch['events_input_ids'])
        padded_events_input_ids = torch.zeros((batch_size, max_events_length), dtype=torch.long)
        padded_events_attention_mask = torch.zeros((batch_size, max_events_length), dtype=torch.long)
        
        for i in range(batch_size):
            events_seq = batch['events_input_ids'][i]
            events_mask = batch['events_attention_mask'][i]
            events_length = len(events_seq)
            
            padded_events_input_ids[i, :events_length] = events_seq
            padded_events_attention_mask[i, :events_length] = events_mask
        
        item_batch_size = len(batch['item_input_ids'])
        max_item_length = max(len(seq) for seq in batch['item_input_ids'])
        
        padded_item_input_ids = torch.zeros((item_batch_size, max_item_length), dtype=torch.long)
        padded_item_attention_mask = torch.zeros((item_batch_size, max_item_length), dtype=torch.long)
        
        for i in range(item_batch_size):
            item_seq = batch['item_input_ids'][i]
            item_mask = batch['item_attention_mask'][i]
            item_length = len(item_seq)
            
            padded_item_input_ids[i, :item_length] = item_seq
            padded_item_attention_mask[i, :item_length] = item_mask
        
        batch['events_input_ids'] = padded_events_input_ids
        batch['events_attention_mask'] = padded_events_attention_mask
        batch['item_input_ids'] = padded_item_input_ids
        batch['item_attention_mask'] = padded_item_attention_mask
        
        return batch


@dataclass
class EmbeddingsCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(examples)
        max_length = max(len(ex['input_ids']) for ex in examples)
        
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        for i, example in enumerate(examples):
            seq_length = len(example['input_ids'])
            input_ids[i, :seq_length] = example['input_ids']
            attention_mask[i, :seq_length] = example['attention_mask']
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        return batch


@dataclass
class UserEmbeddingsCollator(EmbeddingsCollator):
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        batch.update({
            'ids': [ex['id'] for ex in examples],
            'item_ids': [ex['item_ids'] for ex in examples],
            'meta': [ex.get('meta', {}) for ex in examples],
        })
        
        return batch


@dataclass
class ItemEmbeddingsCollator(EmbeddingsCollator):
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        batch.update({
            'ids': [ex['id'] for ex in examples],
            'meta': [ex.get('meta', {}) for ex in examples],
        })
        
        return batch 

@dataclass
class TimeAwareRecommendationDataCollator(RecommendationDataCollator):
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        time_vectors = [example['time_vectors'] for example in examples]
        time_dim = time_vectors[0].size(-1)
        
        batch_size = len(examples)
        max_events_length = batch['events_input_ids'].size(1) 
        
        padded_time_vectors = -1 * torch.ones((batch_size, max_events_length, time_dim), dtype=torch.float32)
        
        for i, time_vecs in enumerate(time_vectors):
            events_length = len(time_vecs)
            padded_time_vectors[i, :events_length, :] = time_vecs
        
        batch['time_vectors'] = padded_time_vectors
        return batch

@dataclass
class TimeAwareUserEmbeddingsCollator(UserEmbeddingsCollator):
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        
        time_vectors = [example['time_vectors'] for example in examples]
        time_dim = time_vectors[0].size(-1)
        batch_size = batch['input_ids'].size(0)
        max_length = batch['input_ids'].size(1)
        
        padded_time_vectors = -1 * torch.ones((batch_size, max_length, time_dim), dtype=torch.float32)
        
        for i, time_vecs in enumerate(time_vectors):
            seq_length = len(time_vecs)
            padded_time_vectors[i, :seq_length, :] = time_vecs
        
        batch['time_vectors'] = padded_time_vectors
        
        return batch