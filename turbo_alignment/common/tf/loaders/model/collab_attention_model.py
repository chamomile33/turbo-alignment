import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from transformers import Qwen2Config, AutoConfig, AutoModel
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2DecoderLayer, 
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward
)
import logging
import deepspeed 
from deepspeed import zero


logger = logging.getLogger(__name__)


class CollabAttentionConfig(Qwen2Config):
    model_type = "collab_attention"
    
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
        self.attention_fusion_type = attention_fusion_type
        self._attn_implementation_autoset = False
        self.attn_implementation = "flash_attention_2"
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "CollabAttentionConfig":
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


class CollabAttention(Qwen2Attention):
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.collab_q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.collab_k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.collab_v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.collab_o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        if config.attention_fusion_type == "gating":
            self.gate_q = nn.Linear(config.hidden_size * 2, config.num_attention_heads * self.head_dim, bias=True)
            self.gate_k = nn.Linear(config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=True)
            self.gate_v = nn.Linear(config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=True)

        self.collab_weight_q = torch.nn.Parameter(torch.zeros(1))
        self.collab_weight_k = torch.nn.Parameter(torch.zeros(1))
        self.collab_weight_v = torch.nn.Parameter(torch.zeros(1))

    def _fuse_projections(
        self,
        hidden_states: torch.Tensor,
        collab_embeddings: torch.Tensor,
        orig_proj: nn.Linear,
        collab_proj: nn.Linear,
        weight: Optional[nn.Parameter] = None,
        gate_proj: Optional[nn.Linear] = None
    ) -> torch.Tensor:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        orig_output = orig_proj(hidden_states)
        collab_output = collab_proj(collab_embeddings)
        
        if self.config.attention_fusion_type == "sum":
            return (orig_output + weight * collab_output).view(hidden_shape).transpose(1, 2)
        
        elif self.config.attention_fusion_type == "gating":
            gate_input = torch.cat([hidden_states, collab_embeddings], dim=-1)
            gate = torch.sigmoid(gate_proj(gate_input))
            return (gate * orig_output + (1 - gate) * weight * collab_output).view(hidden_shape).transpose(1, 2)
        else:
            raise ValueError(f"Неизвестный тип: {self.config.attention_fusion_type}") 


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        collab_hidden_states: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        input_shape = hidden_states.shape[:-1]
        
        if self.config.attention_fusion_type == "gating":
            query_states = self._fuse_projections(
                hidden_states, collab_hidden_states, self.q_proj, self.collab_q_proj, self.collab_weight_q, self.gate_q
            )
            key_states = self._fuse_projections(
                hidden_states, collab_hidden_states, self.k_proj, self.collab_k_proj, self.collab_weight_k, self.gate_k
            )
            value_states = self._fuse_projections(
                hidden_states, collab_hidden_states, self.v_proj, self.collab_v_proj, self.collab_weight_v, self.gate_v
            )
        else:
            query_states = self._fuse_projections(
                hidden_states, collab_hidden_states, self.q_proj, self.collab_q_proj, self.collab_weight_q
            )
            key_states = self._fuse_projections(
                hidden_states, collab_hidden_states, self.k_proj, self.collab_k_proj, self.collab_weight_k
            )
            value_states = self._fuse_projections(
                hidden_states, collab_hidden_states, self.v_proj, self.collab_v_proj, self.collab_weight_v
            )
            
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states,key_states, cos=cos, sin=sin)
        
        if past_key_value is not None:
            cache_kwargs = {}
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights
    

class CollabAttentionDecoderLayer(Qwen2DecoderLayer):
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        
        self.self_attn = CollabAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)   
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        collab_hidden_states: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
       
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
       
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            collab_hidden_states=collab_hidden_states,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs


class CollabAttentionModel(Qwen2PreTrainedModel):
    config_class = CollabAttentionConfig
    base_model_prefix = "collab_attention"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    
    def __init__(self, config: CollabAttentionConfig):
        
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

        for i in range(1, self.num_collaborative_layers + 1):
            layer_idx = len(self.inner_model.layers) - i
            new_layer = CollabAttentionDecoderLayer(self.inner_model.config, layer_idx)
            self.inner_model.layers[layer_idx] = new_layer
        
        self.collaborative_embedding_dim = config.collaborative_embedding_dim
        hidden_dim = int(config.projector_hidden_ratio * config.collaborative_embedding_dim)
        output_dim = config.hidden_size
        
        self._create_and_init_projector_weights(self.collaborative_embedding_dim, hidden_dim , output_dim)


    def replace_layers_with_collab_attention(self):

        for i in range(1, self.num_collaborative_layers + 1):
            layer_idx = len(self.inner_model.layers) - i
            layer = self.inner_model.layers[layer_idx]

            with deepspeed.zero.Init(config_dict_or_path='configs/deepspeed/stage3.json'):
                self.config._attn_implementation = 'flash_attention_2'
                new_layer = CollabAttentionDecoderLayer(self.config, layer_idx)

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

            self.inner_model.layers[layer_idx] = new_layer
            deepspeed.comm.barrier()

    def gradient_checkpointing_enable(self, **kwargs):
        super().gradient_checkpointing_enable(**kwargs)
        self.inner_model.gradient_checkpointing_enable(**kwargs)
    
    def get_input_embeddings(self):
        return self.inner_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.inner_model.set_input_embeddings(value)

    def _create_and_init_projector_weights(self, input_dim, hidden_dim, output_dim):
        with deepspeed.zero.Init(config_dict_or_path='configs/deepspeed/stage3.json'):
            self.embedding_projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim) 
            )
        params_to_gather = list(self.embedding_projector.parameters())

        with zero.GatheredParameters(params_to_gather, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                for module in self.embedding_projector.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
        deepspeed.comm.barrier()
    

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        collab_embeddings: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        if position_ids is None:
            device = input_ids.device
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        collab_embeddings = collab_embeddings.to(dtype=self.embedding_projector[0].weight.dtype)
        collab_embeddings = self.embedding_projector(collab_embeddings)
        
        outputs = self.inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache = False,
            collab_hidden_states=collab_embeddings,
            **kwargs
        )
        
        return outputs

