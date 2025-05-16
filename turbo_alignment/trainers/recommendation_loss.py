import torch
import torch.nn.functional as F
from typing import Literal, Optional, List

from transformers import PreTrainedModel
from accelerate import Accelerator
from turbo_alignment.common.tf.loaders.model import ModelWithMlp

class RecommendationLoss:
    def __init__(
        self,
        pooling_strategy: Literal["mean", "last"] = "mean",
        temperature: float = 1,
        accelerator: Optional[Accelerator] = None,
        gather_items_in_batch: bool = True,
    ):
        self.pooling_strategy = pooling_strategy
        self.temperature = temperature
        self.accelerator = accelerator
        self.gather_items_in_batch = gather_items_in_batch

    def pad_items_and_gather(self, item_embeddings, item_ids, accelerator):

        item_embeddings = accelerator.pad_across_processes(item_embeddings, dim=0)
        item_ids  = accelerator.pad_across_processes(item_ids, dim=0, pad_index=-1)

        all_embeddings = accelerator.gather(item_embeddings)
        all_ids = accelerator.gather(item_ids)
        valid_mask = all_ids != -1
        all_embeddings = all_embeddings[valid_mask]
        all_ids = all_ids[valid_mask]

        return all_embeddings, all_ids


    def __call__(
        self,
        model: PreTrainedModel,
        events_input_ids: torch.Tensor,
        events_attention_mask: torch.Tensor,
        item_input_ids: torch.Tensor,
        item_attention_mask: torch.Tensor,
        item_ids: List[str] = None,
        target_item_ids: List[List[str]] = None,
        time_vectors: torch.Tensor = None,
        events_collab_embeddings: torch.Tensor = None,
        item_collab_embeddings: torch.Tensor = None,
        accelerator: Optional[Accelerator] = None,
        collab_embed_token_id: int = None,
        cross_attention_mask: torch.Tensor = None,
        cross_attention_embeddings: torch.Tensor = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        
        accelerator = accelerator or self.accelerator
        
        wrapped_model = getattr(model, 'module', model)
        is_mlp_wrapper = isinstance(wrapped_model, ModelWithMlp)
        if is_mlp_wrapper:

            events_outputs = model(
                input_ids=events_input_ids,
                attention_mask=events_attention_mask,
                time_vectors = time_vectors,
                collab_embeddings = events_collab_embeddings,
                collab_embed_token_id = collab_embed_token_id,
                encoder_hidden_states = cross_attention_embeddings,
                cross_attention_mask = cross_attention_mask,
            )
            
            item_outputs = model(
                input_ids=item_input_ids,
                attention_mask=item_attention_mask,
                collab_embeddings = item_collab_embeddings,
                collab_embed_token_id = collab_embed_token_id,
            )
            events_embeddings = events_outputs.last_hidden_state.squeeze(1)
            item_embeddings = item_outputs.last_hidden_state.squeeze(1)
        else:
            events_outputs = model(
                input_ids=events_input_ids,
                attention_mask=events_attention_mask,
                output_hidden_states=True,
                time_vectors = time_vectors,
                collab_embeddings = events_collab_embeddings,
                collab_embed_token_id = collab_embed_token_id,
                encoder_hidden_states = cross_attention_embeddings,
                cross_attention_mask = cross_attention_mask
            )
          
            item_outputs = model(
                input_ids=item_input_ids,
                attention_mask=item_attention_mask,
                output_hidden_states=True,
                collab_embeddings = item_collab_embeddings,
                collab_embed_token_id = collab_embed_token_id,
            )

            events_hidden = events_outputs.hidden_states[-1] 
            item_hidden = item_outputs.hidden_states[-1] 
            
            if self.pooling_strategy == "mean":
                events_mask = events_attention_mask.unsqueeze(-1) 
                events_embeddings = (events_hidden * events_mask).sum(dim=1) / events_mask.sum(dim=1)
                
                item_mask = item_attention_mask.unsqueeze(-1) 
                item_embeddings = (item_hidden * item_mask).sum(dim=1) / item_mask.sum(dim=1)
            else:
                batch_size = events_hidden.size(0)
            
                events_lengths = events_attention_mask.sum(dim=1) - 1 
                events_embeddings = torch.stack([
                    events_hidden[i, events_lengths[i]] for i in range(batch_size)
                ])
                
                batch_size = item_hidden.size(0)
                item_lengths = item_attention_mask.sum(dim=1) - 1  
                item_embeddings = torch.stack([
                    item_hidden[i, item_lengths[i]] for i in range(batch_size)
                ])
        
        events_embeddings = F.normalize(events_embeddings, p=2, dim=1)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        
        device = events_embeddings.device
        local_batch_size = events_embeddings.shape[0]
        
        if self.gather_items_in_batch:
            all_item_embeddings, all_item_ids = self.pad_items_and_gather(item_embeddings, item_ids, accelerator)

        else:
            all_item_embeddings = item_embeddings
            all_item_ids = item_ids
        
        all_item_ids = all_item_ids.detach().cpu().numpy()
        unique_indices = []
        unique_item_ids = []
        unique_item_id_to_index = {}
        
        full_target_item_ids = []
        for i, targets in enumerate(target_item_ids):
            full_target_item_ids.extend(targets)
        
        r = 0
        if self.gather_items_in_batch:
            idx_perm = torch.randperm(len(all_item_ids))
            all_item_ids = all_item_ids[idx_perm]
            all_item_embeddings = all_item_embeddings[idx_perm]

        for idx, item_id in enumerate(all_item_ids):
            if item_id not in unique_item_id_to_index:
                if item_id in full_target_item_ids or r%7 == 0:
                    unique_item_id_to_index[item_id] = len(unique_item_ids)
                    unique_indices.append(idx)
                    unique_item_ids.append(item_id)
                    if r%7 == 0:
                        r = 1
                else:
                    r += 1

        unique_item_embeddings = all_item_embeddings[unique_indices]
        similarity = torch.matmul(events_embeddings, unique_item_embeddings.transpose(0, 1)) / self.temperature
        soft_labels = torch.zeros_like(similarity)
        
        for i, targets in enumerate(target_item_ids):
            relevant_indices = []
            for target_id in set(targets):
                target_idx = unique_item_id_to_index[target_id]
                relevant_indices.append(target_idx)
            
            prob_value = 1.0 / len(relevant_indices)
            for idx in relevant_indices:
                soft_labels[i, idx] = prob_value

        log_q = F.log_softmax(similarity, dim=1)
        loss = F.kl_div(log_q, soft_labels, reduction='batchmean')
        
        if accelerator.is_main_process:
            print('Similarity shape', similarity.shape)

        embeddings_dict = {
            "events_embeddings": events_embeddings,
            "item_embeddings": item_embeddings,
            "similarity": similarity,
        }

        
        return loss, embeddings_dict

