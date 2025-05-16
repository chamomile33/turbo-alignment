from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM
import torch.nn as nn
import deepspeed 
from deepspeed import zero

class CollabEmbedsConfig(PretrainedConfig):
    model_type = "collab-embeds-model"

    def __init__(self,
                 base_model_name: str = "",
                 embed_dim: int = 300,
                 projector_hidden_dim: int = 1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.embed_dim = embed_dim
        self.projector_hidden_dim = projector_hidden_dim
        self._attn_implementation_autoset = False
        self.attn_implementation = "flash_attention_2"
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "CollabEmbedsConfig":
        result = super().from_dict(config_dict, **kwargs)
        if isinstance(result, tuple):
            config,_ = result
            config._attn_implementation_autoset = False
            config.attn_implementation = "flash_attention_2"
            return config, _
        else:
            config = result
            config._attn_implementation_autoset = False
            config.attn_implementation = "flash_attention_2"
            return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, **kwargs):
        result = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if isinstance(result, tuple):
            config,_ = result
            config._attn_implementation_autoset = False
            config.attn_implementation = "flash_attention_2"
            return config, _
        else:
            config = result
            config._attn_implementation_autoset = False
            config.attn_implementation = "flash_attention_2"
            return config

class CollaborativeEmbeddingsModel(PreTrainedModel):
    config_class = CollabEmbedsConfig
    base_model_prefix = "custom"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def gradient_checkpointing_enable(self, **kwargs):
        super().gradient_checkpointing_enable(**kwargs)
        self.inner_model.gradient_checkpointing_enable(**kwargs)
    
    def get_input_embeddings(self):
        return self.inner_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.inner_model.set_input_embeddings(value)
        
    def __init__(self, config: CollabEmbedsConfig):
        super().__init__(config)
        config._attn_implementation_autoset = False
        config.attn_implementation = "flash_attention_2"
        config.base_model_name = 'RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4'
        inner_conf = AutoConfig.from_pretrained(config.base_model_name)
        dict_for_update = {key: val for key, val in config.to_dict().items() if key not in ['_name_or_path', 'architectures']}
        inner_conf.update(dict_for_update)
        self.inner_model = AutoModelForCausalLM.from_config(inner_conf)
        self.embed_dim = config.embed_dim

        with deepspeed.zero.Init(config_dict_or_path='configs/deepspeed/stage3.json'):
            self.projector = nn.Sequential(
                nn.Linear(self.embed_dim, config.projector_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.projector_hidden_dim),
                nn.Linear(config.projector_hidden_dim, inner_conf.hidden_size)
            )

    def init_mlp_weights(self):
        params_to_gather = list(self.projector.parameters())
        with zero.GatheredParameters(params_to_gather, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                for module in self.projector.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
        deepspeed.comm.barrier()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        collab_embeddings=None,  
        collab_embed_token_id=None,  
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if collab_embeddings is None:
            raise ValueError("collab_embeddings должны быть предоставлены для CollaborativeEmbeddingsModel")
        if collab_embed_token_id is None:
            raise ValueError("collab_embed_token_id должен быть предоставлен для CollaborativeEmbeddingsModel")
        
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.inner_model.get_input_embeddings()(input_ids)
       
        collab_embeddings = collab_embeddings.to(self.projector[0].weight.dtype)
        projected_collab_embeds = self.projector(collab_embeddings)
        batch_size = input_ids.size(0)
        modified_inputs_embeds = inputs_embeds
    
        for batch_idx in range(batch_size):
            collab_mask = (input_ids[batch_idx] == collab_embed_token_id)
            num_collab_tokens = collab_mask.sum().item()
            
            if num_collab_tokens > 0:
                collab_positions = collab_mask.nonzero(as_tuple=True)[0]
                batch_collab_embeds = projected_collab_embeds[batch_idx, :num_collab_tokens]
                
                inverse_mask = (~collab_mask).unsqueeze(-1).to(modified_inputs_embeds.dtype)
                modified_inputs_embeds[batch_idx] = modified_inputs_embeds[batch_idx] * inverse_mask
                modified_inputs_embeds[batch_idx, collab_positions] += batch_collab_embeds
            
            outputs = self.inner_model.model(
                input_ids=None,  
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=modified_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
            
            return outputs

