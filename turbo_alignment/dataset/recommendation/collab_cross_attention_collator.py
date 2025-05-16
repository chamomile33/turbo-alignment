from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase

from turbo_alignment.dataset.recommendation.cross_attention_collator import CrossAttentionCollatorMixin
from turbo_alignment.dataset.recommendation.collab_attention_collator import CollaborativeAttentionRecommendationDataCollator, CollaborativeAttentionUserCollator
from turbo_alignment.dataset.recommendation.cross_attention_collator import GLOBAL_DYNAMIC_EMBEDDINGS_DICT

@dataclass
class CollabCrossAttentionUserEmbeddingsCollator(CollaborativeAttentionUserCollator, CrossAttentionCollatorMixin):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        global GLOBAL_DYNAMIC_EMBEDDINGS_DICT
        super().__init__(tokenizer)
        self.embeddings_dict = GLOBAL_DYNAMIC_EMBEDDINGS_DICT
   
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
class CollabCrossAttentionRecommendationDataCollator(CollaborativeAttentionRecommendationDataCollator, CrossAttentionCollatorMixin):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        global GLOBAL_DYNAMIC_EMBEDDINGS_DICT
        super().__init__(tokenizer)
        self.embeddings_dict = GLOBAL_DYNAMIC_EMBEDDINGS_DICT
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        for example in examples:
            client_id = example['meta']['client_id']
            target_timestamp = example['meta']['timestamp']
            example['cross_attention_embeddings'] = self._get_user_embeddings(client_id, target_timestamp)
            
        batch = super().__call__(examples)
        cross_attention_data = self.add_cross_attn_embeddings(examples)
        batch.update(cross_attention_data)
        
        return batch