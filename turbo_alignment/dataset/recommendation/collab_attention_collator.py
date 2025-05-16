from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from pathlib import Path 
import torch
from turbo_alignment.common.logging import get_project_logger

from turbo_alignment.dataset.recommendation.collators import (
    RecommendationDataCollator,
    UserEmbeddingsCollator,
    ItemEmbeddingsCollator
)

logger = get_project_logger()


def load_embeddings_dicts():
    global GLOBAL_EMBEDDINGS_DICTS
    embeddings_dir = 'ecomm_embeddings'
    GLOBAL_EMBEDDINGS_DICTS = {}

    items = pd.read_parquet(Path(embeddings_dir) / 'items.pq')
    items = items.apply(lambda x: (x['item']['item_id'] + '$' + x['item']['partner_id'], x['embeddings']), axis = 1)
    GLOBAL_EMBEDDINGS_DICTS['item'] = {k: v for k, v in items}
    del items

    merchants = pd.read_parquet(Path(embeddings_dir)/'merchants.pq')
    merchants = merchants.apply(lambda x: (x['merchant'], x['embedding']), axis = 1)
    GLOBAL_EMBEDDINGS_DICTS['merchant'] = {k: v for k, v in merchants}  
    del merchants

    events = pd.read_parquet(Path(embeddings_dir)/'events.pq')
    events = events.apply(lambda x: (x['event_type'], x['embedding']), axis = 1)
    GLOBAL_EMBEDDINGS_DICTS['event'] = {k: v for k, v in events}  
    del events

    users = pd.read_parquet(Path(embeddings_dir)/'users.pq')
    users = users.apply(lambda x: (x['client_id'] + '$' + str(x['timestamp']), x['embeddings']), axis = 1)
    GLOBAL_EMBEDDINGS_DICTS['user'] = {k: v for k, v in users}  
    del users 

    return GLOBAL_EMBEDDINGS_DICTS


GLOBAL_EMBEDDINGS_DICTS = None
GLOBAL_EMBEDDINGS_DICTS = load_embeddings_dicts()

@dataclass
class CollaborativeAttentionCollatorMixin:

    def __post_init__(self):
        global GLOBAL_EMBEDDINGS_DICTS
        self.collab_embed_token = "<collab-embed>"
        self.embeddings_dicts = GLOBAL_EMBEDDINGS_DICTS
        if self.collab_embed_token not in self.tokenizer.get_vocab():
            logger.warning(f"Токен {self.collab_embed_token} не найден в словаре токенизатора. ")
        
        for emb_dict in self.embeddings_dicts.values():
            if emb_dict:
                first_key = next(iter(emb_dict))
                self.embedding_dim = emb_dict[first_key].shape[-1]
                break
    
        self.collab_embed_token_id = self.tokenizer.convert_tokens_to_ids(self.collab_embed_token)

    def _get_embedding(self, embed_type: str, embed_key: str) -> torch.Tensor:
        if embed_type == 'spec':
            return torch.zeros(self.embedding_dim)
        if embed_type not in self.embeddings_dicts:
            raise Exception(f"Неизвестный тип эмбеддинга: {embed_type}")
        
        if embed_key not in self.embeddings_dicts[embed_type]:
            raise Exception(f"Ключ {embed_key} не найден в словаре эмбеддингов типа {embed_type}")
        
        return torch.tensor(self.embeddings_dicts[embed_type][embed_key])
    
    def process_collaborative_embeddings(
        self, 
        input_ids: list, 
        collaborative_keys: List[Dict[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        
        embed_token_mask = (input_ids_tensor == self.collab_embed_token_id)
        embed_token_indices = embed_token_mask.nonzero().squeeze(-1)
        
        filter_mask = ~embed_token_mask
        
        seq_len = len(input_ids)
        collaborative_embeddings = torch.zeros((seq_len, self.embedding_dim), dtype=torch.float32)
        
        embeddings = torch.stack([
            self._get_embedding(embed_info['type'], embed_info['key'])
            for embed_info in collaborative_keys
        ])
        
        segment_indices = torch.zeros(seq_len, dtype=torch.long)
        
        token_markers = torch.zeros(seq_len, dtype=torch.long)
        token_markers[embed_token_indices] = 1
    
        segment_indices = token_markers.cumsum(0)
        segment_indices = segment_indices - 1
        collaborative_embeddings = embeddings[segment_indices]
    
        filtered_input_ids = input_ids_tensor[filter_mask]
        filtered_collaborative_embeddings = collaborative_embeddings[filter_mask]
        
        attention_mask = torch.ones_like(filtered_input_ids)
        
        return filtered_input_ids, attention_mask, filtered_collaborative_embeddings
    
    def pad_embeddings(
        self, 
        embeddings: List[torch.Tensor], 
    ) -> torch.Tensor:
        
        batch_size = len(embeddings)
        max_length = max(emb.size(0) for emb in embeddings)
        embedding_dim = embeddings[0].size(-1)
        
        padded_embeddings = torch.zeros(
            (batch_size, max_length, embedding_dim), 
            dtype=torch.float32
        )
        for i, embed in enumerate(embeddings):
            seq_len = embed.size(0)
            padded_embeddings[i, :seq_len, :] = embed
        
        return padded_embeddings


@dataclass
class CollaborativeAttentionRecommendationDataCollator(RecommendationDataCollator, CollaborativeAttentionCollatorMixin):

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:

        item_collaborative_embeddings = []
        events_collaborative_embeddings = []

        for example in examples:
            filtered_input_ids, filtered_attention_mask, collaborative_embeddings = self.process_collaborative_embeddings(
                    example['events_input_ids'], 
                    example['events_collaborative_keys']
            )
            example['events_input_ids'] = filtered_input_ids
            example['events_attention_mask'] = filtered_attention_mask
            events_collaborative_embeddings.append(collaborative_embeddings)

            for i,tokenized_item in enumerate(example['tokenized_items']):
                item_input_ids, item_attention_mask, item_embeddings = self.process_collaborative_embeddings(
                    tokenized_item['input_ids'], 
                    tokenized_item['item_collaborative_keys']
                )
                example['tokenized_items'][i]['input_ids'] = item_input_ids
                example['tokenized_items'][i]['attention_mask'] = item_attention_mask
                item_collaborative_embeddings.append(item_embeddings)
        
        batch = super().__call__(examples)
        
        events_padded_embeddings = self.pad_embeddings(events_collaborative_embeddings)
        items_padded_embeddings = self.pad_embeddings(item_collaborative_embeddings)

        batch['events_collab_embeddings'] = events_padded_embeddings
        batch['item_collab_embeddings'] = items_padded_embeddings
        
        return batch


@dataclass
class CollaborativeAttentionUserCollator(UserEmbeddingsCollator, CollaborativeAttentionCollatorMixin):
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        collaborative_embeddings = []

        for example in examples:
            filtered_input_ids, filtered_attention_mask, collab_embeddings = self.process_collaborative_embeddings(
                    example['input_ids'], 
                    example['collaborative_keys']
            )
            example['input_ids'] = filtered_input_ids
            example['attention_mask'] = filtered_attention_mask
            collaborative_embeddings.append(collab_embeddings)

        batch = super().__call__(examples)

        padded_embeddings = self.pad_embeddings(collaborative_embeddings)
        batch['collab_embeddings'] = padded_embeddings
        
        return batch


@dataclass
class CollaborativeAttentionItemCollator(ItemEmbeddingsCollator, CollaborativeAttentionCollatorMixin):
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        collaborative_embeddings = []

        for example in examples:
            filtered_input_ids, filtered_attention_mask, collab_embeddings = self.process_collaborative_embeddings(
                    example['input_ids'], 
                    example['collaborative_keys']
            )
            example['input_ids'] = filtered_input_ids
            example['attention_mask'] = filtered_attention_mask
            collaborative_embeddings.append(collab_embeddings)

        batch = super().__call__(examples)

        padded_embeddings = self.pad_embeddings(collaborative_embeddings)
        batch['collab_embeddings'] = padded_embeddings
        
        return batch 