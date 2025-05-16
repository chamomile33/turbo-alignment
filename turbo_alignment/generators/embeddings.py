from typing import Any, Optional, TypeVar, Generic

import torch
import numpy as np
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.embeddings import EmbeddingsGenerationSettings
from turbo_alignment.settings.generators.outputs.embeddings import EmbeddingsInferenceOutput, ItemEmbeddingOutput
from turbo_alignment.dataset.recommendation.models import UserRecommendationRecord, ItemRecord, TimeAwareUserRecommendationRecord, CollabEmbedsUserRecommendationRecord, CollabEmbedsItemRecord
from turbo_alignment.dataset.recommendation.collators import UserEmbeddingsCollator, ItemEmbeddingsCollator, EmbeddingsCollator, TimeAwareUserEmbeddingsCollator
from turbo_alignment.dataset.recommendation.collab_embeds_collator import CollaborativeUserEmbeddingsCollator,CollaborativeItemEmbeddingsCollator
from turbo_alignment.common.tf.loaders.model import ModelWithMlp
from turbo_alignment.common.tf.loaders.model.cross_attention_model import CrossAttentionQwen2Model
from turbo_alignment.common.tf.loaders.model.collab_attention_model import CollabAttentionModel
from turbo_alignment.dataset.recommendation.cross_attention_collator import CrossAttentionUserEmbeddingsCollator
from turbo_alignment.dataset.recommendation.collab_attention_collator import CollaborativeAttentionUserCollator, CollaborativeAttentionItemCollator
from turbo_alignment.common.tf.loaders.model.collab_cross_attention_model import CollabCrossAttentionModel
from turbo_alignment.dataset.recommendation.collab_cross_attention_collator import CollabCrossAttentionUserEmbeddingsCollator

T = TypeVar('T', UserRecommendationRecord, ItemRecord)


class EmbeddingsGenerator(Generic[T], BaseGenerator[T, EmbeddingsInferenceOutput]):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        custom_generation_settings: EmbeddingsGenerationSettings,
        batch: Optional[int] = None,
        accelerator: Optional[Accelerator] = None,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            accelerator=accelerator,
        )
        self._model.eval()
        self._collator = None
        self._custom_generation_settings = custom_generation_settings
        wrapped_model = getattr(model, 'module', model)
        self._is_mlp_wrapper = isinstance(wrapped_model, ModelWithMlp)
        
    def _get_collator(self, records, tokenizer: PreTrainedTokenizerBase) -> EmbeddingsCollator:
        if isinstance(self._model, CollabCrossAttentionModel) and self.__orig_class__.__args__[0] == CollabEmbedsUserRecommendationRecord:
            return CollabCrossAttentionUserEmbeddingsCollator(tokenizer=tokenizer)
        elif isinstance(self._model, CollabCrossAttentionModel) and self.__orig_class__.__args__[0] == CollabEmbedsItemRecord:
            return CollaborativeAttentionItemCollator(tokenizer=tokenizer)
        elif isinstance(self._model, CollabAttentionModel) and self.__orig_class__.__args__[0] == CollabEmbedsUserRecommendationRecord:
            return CollaborativeAttentionUserCollator(tokenizer=tokenizer)
        elif isinstance(self._model, CollabAttentionModel) and self.__orig_class__.__args__[0] == CollabEmbedsItemRecord:
            return CollaborativeAttentionItemCollator(tokenizer=tokenizer)
        elif isinstance(self._model, CrossAttentionQwen2Model) and self.__orig_class__.__args__[0] == UserRecommendationRecord:
            return CrossAttentionUserEmbeddingsCollator(tokenizer=tokenizer)
        elif self.__orig_class__.__args__[0] == TimeAwareUserRecommendationRecord:
            return TimeAwareUserEmbeddingsCollator(tokenizer=tokenizer)
        elif self.__orig_class__.__args__[0] == CollabEmbedsUserRecommendationRecord:
            return CollaborativeUserEmbeddingsCollator(tokenizer=tokenizer)
        elif self.__orig_class__.__args__[0] == CollabEmbedsItemRecord:
            return CollaborativeItemEmbeddingsCollator(tokenizer=tokenizer)
        elif self.__orig_class__.__args__[0] == UserRecommendationRecord:
            return UserEmbeddingsCollator(tokenizer=tokenizer)
        elif self.__orig_class__.__args__[0] == ItemRecord:
            return ItemEmbeddingsCollator(tokenizer=tokenizer)

    def _generate_from_batch(
        self,
        records: list[dict[str, torch.Tensor]],
        original_records: list[T],
        dataset_name: str,
    ) -> list[EmbeddingsInferenceOutput]:
        if self._collator is None:
            self._collator = self._get_collator(records, self._tokenizer)
        
        batch = self._collator(records)
        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            if 'time_vectors' in batch:
                inputs["time_vectors"] = batch["time_vectors"].to(self.device)
            if 'collab_embeddings' in batch:
                inputs["collab_embeddings"] = batch["collab_embeddings"].to(self.device)
            if 'collab_embed_token_id' in batch:
                inputs['collab_embed_token_id'] = batch['collab_embed_token_id']
            if 'cross_attention_embeddings' in batch:
                inputs['encoder_hidden_states'] = batch['cross_attention_embeddings'].to(self.device)
                inputs['cross_attention_mask'] = batch['cross_attention_mask'].to(self.device)
            
            if not self._is_mlp_wrapper:
                inputs["output_hidden_states"] = True
                inputs["return_dict"] = True
            
            outputs = self._model(**inputs)
            
            if self._is_mlp_wrapper:
                embeddings = outputs.last_hidden_state.float().squeeze(1).cpu().numpy()
            else:
                hidden_states = outputs.hidden_states[-1] 
                pooling_strategy = getattr(self._custom_generation_settings, 'pooling_strategy', 'mean')
                
                if pooling_strategy == 'mean':
                    embeddings = self._mean_pooling(hidden_states, batch["attention_mask"].to(self.device)).cpu().numpy()
                elif pooling_strategy == 'last':
                    embeddings = self._last_token_pooling(hidden_states, batch["attention_mask"].to(self.device)).float().cpu().numpy()
            
            embeddings = self._normalize_embeddings(embeddings)

        return [
            self._create_output(record, embedding, dataset_name)
            for record, embedding in zip(original_records, embeddings)
        ]

    def _generate_from_single_record(
        self,
        record: dict[str, Any],
        original_record: T,
        dataset_name: str,
    ) -> EmbeddingsInferenceOutput:
        input_ids = record['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = record['attention_mask'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if 'time_vectors' in record:
                inputs["time_vectors"] = record["time_vectors"].unsqueeze(0).to(self.device)
            if 'collab_embeddings' in record:
                inputs["collab_embeddings"] = record["collab_embeddings"].unsqueeze(0).to(self.device)
            if 'collab_embed_token_id' in record:
                inputs['collab_embed_token_id'] = record['collab_embed_token_id']
            if 'cross_attention_embeddings' in record:
                inputs['encoder_hidden_states'] = record['cross_attention_embeddings'].unsqueeze(0).to(self.device)
                inputs['cross_attention_mask'] = record['cross_attention_mask'].unsqueeze(0).to(self.device)

            if not self._is_mlp_wrapper:
                inputs["output_hidden_states"] = True
                inputs["return_dict"] = True
            
            outputs = self._model(**inputs)
            
            if self._is_mlp_wrapper:
                embedding = outputs.last_hidden_state.squeeze().cpu().numpy()
            else:
                hidden_states = outputs.hidden_states[-1] 
                pooling_strategy = getattr(self._custom_generation_settings, 'pooling_strategy', 'mean')
            
                if pooling_strategy == 'mean':
                    embedding = self._mean_pooling(hidden_states, attention_mask).cpu().numpy()[0]
                elif pooling_strategy == 'last':
                    embedding = self._last_token_pooling(hidden_states, attention_mask).cpu().numpy()[0]

            embedding = self._normalize_embeddings(embedding)
        
        return self._create_output(original_record, embedding, dataset_name)
    
    def _create_output(self, record: T, embedding: np.ndarray, dataset_name: str) -> EmbeddingsInferenceOutput:
        if isinstance(record, UserRecommendationRecord):
            return EmbeddingsInferenceOutput(
                id=record.id,
                dataset_name=dataset_name,
                embedding=embedding,
                item_ids=record.item_ids,
                meta=record.meta,
            )
        elif isinstance(record, ItemRecord):
            return ItemEmbeddingOutput(
                id=record.id,
                item_id=record.item_id,
                dataset_name=dataset_name,
                embedding=embedding,
                meta=record.meta,
                item_text=record.item_text,
            )
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1)
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        return sum_embeddings / torch.clamp(sum_mask.float(), min=1e-8)
    
    def _last_token_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = token_embeddings.size(0)

        last_token_indices = attention_mask.sum(dim=1) - 1 
        
        last_token_embeddings = torch.stack([
            token_embeddings[i, last_token_indices[i]] 
            for i in range(batch_size)
        ]) 
        
        return last_token_embeddings
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings.shape) == 1:
            norm = np.linalg.norm(embeddings)
            return embeddings / max(norm, 1e-8)
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.maximum(norms, 1e-8) 