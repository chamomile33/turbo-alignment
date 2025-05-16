import torch
import torch.nn as nn
from typing import Optional
from transformers import Qwen2Config, AutoConfig, AutoModel
from transformers.cache_utils import Cache
import logging
import deepspeed 
from deepspeed import zero
from turbo_alignment.common.tf.loaders.model.cross_attention_model import CrossQwen2DecoderLayer, CrossAttentionQwen2Model
from turbo_alignment.common.tf.loaders.model.collab_attention_model import CollabAttention, CollabAttentionConfig


logger = logging.getLogger(__name__)


class CollabCrossAttentionConfig(CollabAttentionConfig):
    model_type = "collab_cross_attention"

    def __init__(
        self,
        base_model_name: str = "",
        num_collaborative_layers: int = 6,
        collaborative_embedding_dim: Optional[int] = None,
        projector_hidden_ratio: int = 4,
        attention_fusion_type: str = "sum",  # sum, gating
        **kwargs
    ): 
        super().__init__(**kwargs)
       
        self.base_model_name = base_model_name
        self.num_collaborative_layers = num_collaborative_layers
        self.collaborative_embedding_dim = collaborative_embedding_dim
        self.projector_hidden_ratio = projector_hidden_ratio
        self.num_cross_layers = num_collaborative_layers
        self.attention_fusion_type = attention_fusion_type
        self._attn_implementation_autoset = False
        self.attn_implementation = "flash_attention_2"


class CollabCrossDecoderLayer(CrossQwen2DecoderLayer):
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = CollabAttention(config=config, layer_idx=layer_idx)
        

class CollabCrossAttentionModel(CrossAttentionQwen2Model):
    config_class = CollabCrossAttentionConfig
    base_model_prefix = "collab_cross_attention"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True


    def __init__(self, config: CollabCrossAttentionConfig):
        
        super().__init__(config)
        config._attn_implementation_autoset = False
        config.attn_implementation = "flash_attention_2"
        self.config = config

        config.base_model_name = 'RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4'
        inner_conf = AutoConfig.from_pretrained(config.base_model_name)
        dict_for_update = {key: val for key, val in config.to_dict().items() if key not in ['_name_or_path', 'architectures']}
        inner_conf.update(dict_for_update)
        self.inner_model = AutoModel.from_config(inner_conf)
        self.num_collaborative_layers = min(config.num_collaborative_layers, len(self.inner_model.layers))

        for i in range(1, self.num_cross_layers + 1):
            layer_idx = len(self.inner_model.layers) - i
            self.inner_model.config._attn_implementation = 'flash_attention_2'
            new_layer = CollabCrossDecoderLayer(self.inner_model.config, layer_idx)
            self.inner_model.layers[layer_idx] = new_layer
    
        self.collaborative_embedding_dim = config.collaborative_embedding_dim
        hidden_dim = int(config.projector_hidden_ratio * config.collaborative_embedding_dim)
        output_dim = config.hidden_size
            
        self.collab_projector = self._create_and_init_projector_weights(self.collaborative_embedding_dim, hidden_dim , output_dim)
        self.embedding_projector = self._create_and_init_projector_weights(self.collaborative_embedding_dim, hidden_dim , output_dim)


    def _create_and_init_projector_weights(self, input_dim, hidden_dim, output_dim):
            with deepspeed.zero.Init(config_dict_or_path='configs/deepspeed/stage3.json'):
                embedding_projector = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim) 
                )
            params_to_gather = list(embedding_projector.parameters())

            with zero.GatheredParameters(params_to_gather, modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    for module in embedding_projector.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.normal_(module.weight, mean=0.0, std=0.02)
                            if module.bias is not None:
                                nn.init.zeros_(module.bias)
            deepspeed.comm.barrier()
            return embedding_projector


    def replace_layers_with_collab_cross_attention(self):

        for i in range(1, self.num_cross_layers + 1):
            layer_idx = len(self.inner_model.layers) - i
            layer = self.inner_model.layers[layer_idx]

            with deepspeed.zero.Init(config_dict_or_path='configs/deepspeed/stage3.json'):
                self.config._attn_implementation = 'flash_attention_2'
                new_layer = CollabCrossDecoderLayer(self.config, layer_idx)

            params_to_gather = list(layer.parameters()) + list(new_layer.parameters())
            with zero.GatheredParameters(params_to_gather, modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    new_layer.load_state_dict(layer.state_dict(), strict=False)

                    new_layer.self_attn.collab_q_proj.weight.data.normal_(mean=0.0, std=0.02)
                    new_layer.self_attn.collab_k_proj.weight.data.normal_(mean=0.0, std=0.02)
                    new_layer.self_attn.collab_v_proj.weight.data.normal_(mean=0.0, std=0.02)
                    new_layer.self_attn.collab_q_proj.bias.data.zero_()
                    new_layer.self_attn.collab_k_proj.bias.data.zero_()
                    new_layer.self_attn.collab_v_proj.bias.data.zero_()
                    if self.config.attention_fusion_type == 'gating':
                        self._init_weights(new_layer.self_attn.gate_q)
                        self._init_weights(new_layer.self_attn.gate_k)
                        self._init_weights(new_layer.self_attn.gate_v)

                    new_layer.cross_attn.q_proj.weight.data.normal_(mean=0.0, std=0.02)
                    new_layer.cross_attn.k_proj.weight.data.normal_(mean=0.0, std=0.02)
                    new_layer.cross_attn.v_proj.weight.data.normal_(mean=0.0, std=0.02)
                    new_layer.cross_attn.o_proj.weight.data.normal_(mean=0.0, std=0.02)
                    new_layer.cross_attn.v_proj.bias.data.zero_()
                    new_layer.cross_attn.k_proj.bias.data.zero_()
                    new_layer.cross_attn.q_proj.bias.data.zero_()
                    new_layer.post_cross_attention_layernorm.weight.data.fill_(1.0)
                   

            self.inner_model.layers[layer_idx] = new_layer
            deepspeed.comm.barrier()


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            collab_embeddings: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
        ):
            collab_embeddings = collab_embeddings.to(dtype=self.collab_projector[0].weight.dtype)
            collab_embeddings = self.collab_projector(collab_embeddings)

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                collab_embeddings=collab_embeddings,
                cross_attention_mask=cross_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
            
            return outputs