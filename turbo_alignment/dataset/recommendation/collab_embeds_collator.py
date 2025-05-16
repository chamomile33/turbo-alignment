from dataclasses import dataclass
from typing import Any, Dict, List
import torch

from turbo_alignment.dataset.recommendation.collators import RecommendationDataCollator, UserEmbeddingsCollator, ItemEmbeddingsCollator

@dataclass
class CollaborativeEmbeddingsCollatorMixin:
    def pad_collab_embeddings(
        self, 
        collab_embeddings: List[torch.Tensor], 
        batch_size: int
    ) -> torch.Tensor:
       
        max_embeds = max(len(embeds) for embeds in collab_embeddings) if collab_embeddings else 0
        embed_dim = collab_embeddings[0].size(-1) if max_embeds > 0 else 0
        
        if max_embeds > 0 and embed_dim > 0:
            padded_collab_embeddings = torch.zeros((batch_size, max_embeds, embed_dim), dtype=torch.float32)
            
            for i, embeds in enumerate(collab_embeddings):
                if len(embeds) > 0:
                    padded_collab_embeddings[i, :len(embeds), :] = embeds
            
            return padded_collab_embeddings
        else:
            return torch.tensor([])
    
    def get_collab_embed_token_id(self) -> int:
        collab_embed_token = "<collab-embed>"
        return self.tokenizer.convert_tokens_to_ids(collab_embed_token)


@dataclass
class CollaborativeRecommendationDataCollator(RecommendationDataCollator, CollaborativeEmbeddingsCollatorMixin):
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        collab_embed_token_id = self.get_collab_embed_token_id()
        
        events_collab_embeddings = []
        for example in examples:
            events_collab_embeddings.append(example['events_collab_embeddings'])
        
        batch['events_collab_embeddings'] = self.pad_collab_embeddings(
            events_collab_embeddings, 
            len(examples)
        )
        item_collab_embeddings = []
        for example in examples:
            for item in example['tokenized_items']:
                item_collab_embeddings.append(item['collab_embeddings'])
        
        batch['item_collab_embeddings'] = self.pad_collab_embeddings(
            item_collab_embeddings, 
            len(batch['item_input_ids'])
        )
        batch['collab_embed_token_id'] = collab_embed_token_id
        
        return batch
    

@dataclass
class CollaborativeUserEmbeddingsCollator(UserEmbeddingsCollator, CollaborativeEmbeddingsCollatorMixin):
   
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        collab_embed_token_id = self.get_collab_embed_token_id()
    
        collab_embeddings = []
        for example in examples:
            collab_embeddings.append(example['collab_embeddings'])
    
        batch['collab_embeddings'] = self.pad_collab_embeddings(
            collab_embeddings, 
            batch['input_ids'].size(0)
        )
    
        batch['collab_embed_token_id'] = collab_embed_token_id
        
        return batch


@dataclass
class CollaborativeItemEmbeddingsCollator(ItemEmbeddingsCollator, CollaborativeEmbeddingsCollatorMixin):

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        
        collab_embed_token_id = self.get_collab_embed_token_id()
        collab_embeddings = []
        for example in examples:
            collab_embeddings.append(example['collab_embeddings'])
        
        batch['collab_embeddings'] = self.pad_collab_embeddings(
            collab_embeddings, 
            batch['input_ids'].size(0)
        )
        batch['collab_embed_token_id'] = collab_embed_token_id
        
        return batch