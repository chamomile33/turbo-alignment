import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)

from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import nested_detach

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.trainers.recommendation_loss import RecommendationLoss


class RecommendationTrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        recommendation_loss: Optional[RecommendationLoss] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs,
        )
        self.recommendation_loss = recommendation_loss or RecommendationLoss()
        
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch=None, 
        gather_items_in_batch=True,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        
        events_input_ids = inputs.get("events_input_ids")
        events_attention_mask = inputs.get("events_attention_mask")
        item_input_ids = inputs.get("item_input_ids")
        item_attention_mask = inputs.get("item_attention_mask")
        time_vectors = inputs.get("time_vectors", None)
        events_collab_embeddings = inputs.get("events_collab_embeddings", None)
        item_collab_embeddings = inputs.get("item_collab_embeddings", None)
        collab_embed_token_id = inputs.get("collab_embed_token_id", None)
        cross_attention_embeddings =  inputs.get("cross_attention_embeddings", None)
        cross_attention_mask = inputs.get("cross_attention_mask", None)
       
        item_ids = inputs.get("item_ids", None)
        target_item_ids = inputs.get("target_item_ids", None)
        
        loss, embeddings_dict = self.recommendation_loss(
            model=model,
            events_input_ids=events_input_ids,
            events_attention_mask=events_attention_mask,
            item_input_ids=item_input_ids,
            item_attention_mask=item_attention_mask,
            item_ids=item_ids,
            target_item_ids=target_item_ids,
            time_vectors = time_vectors,
            events_collab_embeddings = events_collab_embeddings,
            item_collab_embeddings=item_collab_embeddings,
            accelerator=self.accelerator,
            collab_embed_token_id = collab_embed_token_id,
            cross_attention_embeddings = cross_attention_embeddings,
            cross_attention_mask = cross_attention_mask,
        )

        loss = loss / self.args.gradient_accumulation_steps
        
        if return_outputs:
            outputs = {
                "loss": loss,
                "events_embeddings": embeddings_dict["events_embeddings"],
                "item_embeddings": embeddings_dict["item_embeddings"],
                "similarity": embeddings_dict["similarity"],
            }
            return loss, outputs
        else:
            return loss
        
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
     
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss *= self.args.gradient_accumulation_steps
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    