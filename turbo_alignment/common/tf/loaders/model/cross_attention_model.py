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
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


class CrossAttentionQwen2Config(Qwen2Config):
    model_type = "cross_attention_qwen2"
    
    def __init__(
        self,
        base_model_name: str = "",
        num_cross_layers: int = 3,
        collaborative_embedding_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
    
        self.base_model_name = base_model_name
        self.num_cross_layers = num_cross_layers
        self.collaborative_embedding_dim = collaborative_embedding_dim
        self._attn_implementation_autoset = False
        self.attn_implementation = "flash_attention_2"
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "CrossAttentionQwen2Config":
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


class CrossAttention(Qwen2Attention):
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            encoder_position_embeddings = position_embeddings
            
        encoder_shape = encoder_hidden_states.shape[:-1]
        encoder_hidden_shape = (*encoder_shape, -1, self.head_dim)
        
        key_states = self.k_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states,torch.zeros_like(query_states), cos=cos, sin=sin)[0]
        
        if encoder_position_embeddings is not None:
            encoder_cos, encoder_sin = encoder_position_embeddings
            key_states = apply_rotary_pos_emb(torch.zeros_like(key_states), key_states, cos = encoder_cos, sin = encoder_sin)[1]
        
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


class CrossQwen2DecoderLayer(Qwen2DecoderLayer):
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.cross_attn = CrossAttention(config=config, layer_idx=layer_idx)
        self.post_cross_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None, 
        encoder_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
       
        if 'collab_embeddings' in kwargs:
            kwargs['collab_hidden_states'] = kwargs['collab_embeddings']

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
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            
            hidden_states, cross_attn_weights = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                position_embeddings=position_embeddings,
                encoder_position_embeddings=encoder_position_embeddings,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)
        
        return outputs


class CrossAttentionQwen2Model(Qwen2PreTrainedModel):

    config_class = CrossAttentionQwen2Config
    base_model_prefix = "cross_attention_qwen2"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    
    def __init__(self, config: CrossAttentionQwen2Config):
        
        super().__init__(config)
        config._attn_implementation_autoset = False
        config.attn_implementation = "flash_attention_2"
        self.config = config

        config.base_model_name = 'RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4'
        inner_conf = AutoConfig.from_pretrained(config.base_model_name)
        dict_for_update = {key: val for key, val in config.to_dict().items() if key not in ['_name_or_path', 'architectures']}
        inner_conf.update(dict_for_update)
        self.inner_model = AutoModel.from_config(inner_conf)
        self.num_cross_layers = min(config.num_cross_layers, len(self.inner_model.layers))

        for i in range(1, self.num_cross_layers + 1):
            layer_idx = len(self.inner_model.layers) - i
            self.inner_model.config._attn_implementation = 'flash_attention_2'
            new_layer = CrossQwen2DecoderLayer(self.inner_model.config, layer_idx)
            self.inner_model.layers[layer_idx] = new_layer

        self.has_embedding_projector = config.collaborative_embedding_dim is not None

        if self.has_embedding_projector:
            self.collaborative_embedding_dim = config.collaborative_embedding_dim
            hidden_dim = 4 * config.collaborative_embedding_dim
            output_dim = config.hidden_size
            self._create_and_init_projector_weights(self.collaborative_embedding_dim, hidden_dim , output_dim)
        
        if hasattr(self.inner_model, "_init_gradient_checkpointing"):
            self.inner_model._init_gradient_checkpointing()


    def replace_layers_with_cross_attention(self):

        for i in range(1, self.num_cross_layers + 1):
            layer_idx = len(self.inner_model.layers) - i
            layer = self.inner_model.layers[layer_idx]

            with deepspeed.zero.Init(config_dict_or_path='configs/deepspeed/stage3.json'):
                self.config._attn_implementation = 'flash_attention_2'
                new_layer = CrossQwen2DecoderLayer(self.inner_model.config, layer_idx)

            params_to_gather = list(layer.parameters()) + list(new_layer.parameters())
            with zero.GatheredParameters(params_to_gather, modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    new_layer.load_state_dict(layer.state_dict(), strict=False)
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
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
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

        if encoder_hidden_states is not None and self.has_embedding_projector:
            encoder_hidden_states = encoder_hidden_states.to(dtype=self.embedding_projector[0].weight.dtype)
            encoder_hidden_states = self.embedding_projector(encoder_hidden_states)
        
        if encoder_hidden_states is not None:
            batch_size = encoder_hidden_states.shape[0]
            encoder_seq_len = encoder_hidden_states.shape[1]
            
            encoder_position_ids = torch.arange(
                0, encoder_seq_len, device=encoder_hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)
            
            encoder_position_embeddings = self.inner_model.rotary_emb(
                encoder_hidden_states, encoder_position_ids
            )
            kwargs["encoder_position_embeddings"] = encoder_position_embeddings
            kwargs["cross_attention_mask"] = cross_attention_mask
              
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
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )
        
        return outputs
