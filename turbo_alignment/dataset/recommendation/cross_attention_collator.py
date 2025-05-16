from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import numpy as np 
import pandas as pd 
import bisect 
from datetime import datetime as dt
from pathlib import Path

from turbo_alignment.dataset.recommendation.collators import RecommendationDataCollator, UserEmbeddingsCollator


def load_dynamic_embeddings_dicts():
    global GLOBAL_DYNAMIC_EMBEDDINGS_DICT
    embeddings_dir = 'ecomm_embeddings'
    GLOBAL_DYNAMIC_EMBEDDINGS_DICT = {}

    users_embeddings = pd.read_parquet(Path(embeddings_dir) / 'users_dynamic_embeddings.pq')
    users_embeddings = users_embeddings.apply(lambda x: (x['client_id'], x['timestamps'], x['embeddings']), axis = 1)
    GLOBAL_DYNAMIC_EMBEDDINGS_DICT = {k: (t, e) for k, t, e in users_embeddings}
    return GLOBAL_DYNAMIC_EMBEDDINGS_DICT


GLOBAL_DYNAMIC_EMBEDDINGS_DICT = None
GLOBAL_DYNAMIC_EMBEDDINGS_DICT = load_dynamic_embeddings_dicts()


class CrossAttentionCollatorMixin:
    def __post_init__(self):
        global GLOBAL_DYNAMIC_EMBEDDINGS_DICT
        self.embeddings_dict = GLOBAL_DYNAMIC_EMBEDDINGS_DICT

    def add_cross_attn_embeddings(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        has_cross_attention = all('cross_attention_embeddings' in ex for ex in examples)
        result = {}
        if has_cross_attention:
            cross_attention_embeddings = [ex['cross_attention_embeddings'] for ex in examples]
          
            batch_size = len(cross_attention_embeddings)
            embed_dim = cross_attention_embeddings[0].size(-1)
            
            max_embeds_seq_len = max(emb.size(0) for emb in cross_attention_embeddings)
            padded_embeddings = torch.zeros((batch_size, max_embeds_seq_len, embed_dim), dtype=torch.float32)
            cross_attention_mask = torch.zeros((batch_size, max_embeds_seq_len), dtype=torch.long)
            
            for i, embeddings in enumerate(cross_attention_embeddings):
                seq_len = embeddings.size(0)
                padded_embeddings[i, :seq_len, :] = embeddings
                cross_attention_mask[i, :seq_len] = 1  
            
            result['cross_attention_embeddings'] = padded_embeddings
            result['cross_attention_mask'] = cross_attention_mask
        
        return result

    def _get_user_embeddings(self, client_id: str, target_timestamp: str) -> torch.Tensor:
        if client_id not in self.embeddings_dict:
            return torch.zeros((1,300))
        
        timestamps, embeddings = self.embeddings_dict[client_id]
        
        if len(timestamps) == 0:
            raise Exception(f"Пустой список таймстемпов для пользователя {client_id}")
        
        target_dt = np.datetime64(dt.fromisoformat(target_timestamp))
        timestamps_dt = pd.to_datetime(timestamps)
        pos_new = bisect.bisect_right(timestamps_dt, target_dt)
        
        if pos_new > 0:
            return torch.tensor(np.stack(embeddings[:pos_new]))
        else:
            return torch.zeros_like(torch.tensor(np.stack(embeddings[:1])))
    
@dataclass
class CrossAttentionUserEmbeddingsCollator(UserEmbeddingsCollator, CrossAttentionCollatorMixin):
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        for example in examples:
            client_id = example['meta']['client_id']
            target_timestamp = example['meta']['timestamp']
            example['cross_attention_embeddings'] = self._get_user_embeddings(client_id, target_timestamp)

        batch = super().__call__(examples)
        cross_attention_data = self.add_cross_attn_embeddings(examples)
        batch.update(cross_attention_data)
        
        return batch

@dataclass
class CrossAttentionRecommendationDataCollator(RecommendationDataCollator, CrossAttentionCollatorMixin):
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        for example in examples:
            client_id = example['meta']['client_id']
            target_timestamp = example['meta']['timestamp']
            example['cross_attention_embeddings'] = self._get_user_embeddings(client_id, target_timestamp)

        batch = super().__call__(examples)
        cross_attention_data = self.add_cross_attn_embeddings(examples)
        batch.update(cross_attention_data)
        
        return batch