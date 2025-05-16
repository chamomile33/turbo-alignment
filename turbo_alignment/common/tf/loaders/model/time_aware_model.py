import torch
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig,AutoModelForCausalLM
import torch.nn as nn
import deepspeed 
from deepspeed import zero

class TimeAwareConfig(PretrainedConfig):
    model_type = "time-aware-model"

    def __init__(self,
                 base_model_name: str = "",
                 time_dim: int = 6,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.time_dim = time_dim
        self._attn_implementation_autoset = False
        self.attn_implementation = "flash_attention_2"
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
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


class TimeAwareModel(PreTrainedModel):
   
    config_class = TimeAwareConfig
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
    
    def __init__(self, config: TimeAwareConfig):
        super().__init__(config)
        config._attn_implementation_autoset = False
        config.attn_implementation = "flash_attention_2"
        config.base_model_name = 'RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4'
        inner_conf = AutoConfig.from_pretrained(config.base_model_name)
        dict_for_update = {key: val for key, val in config.to_dict().items() if key not in ['_name_or_path', 'architectures']}
        inner_conf.update(dict_for_update)
        self.inner_model = AutoModelForCausalLM.from_config(inner_conf)
        self.time_dim = config.time_dim

        with deepspeed.zero.Init(config_dict_or_path='configs/deepspeed/stage3.json'):
            self.fourier_weights = torch.nn.Parameter(
                    torch.zeros(self.time_dim, self.inner_model.config.hidden_size // 2)
                )
                        
            self.fourier_bias = torch.nn.Parameter(
                    torch.zeros(self.inner_model.config.hidden_size // 2)
                )

        self.time_embedding_scale = torch.nn.Parameter(torch.zeros(1))
        self.time_embedding_scale.requires_grad = True

        self.post_init()
    
    def init_fourier(self):
        with zero.GatheredParameters(self.fourier_weights, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                self.fourier_weights.data.normal_(mean=0.0, std=0.5)
                self.fourier_bias.data.normal_(mean=0.0, std=0.5)
        
        deepspeed.comm.barrier()

    def _apply_random_fourier_transform(self, time_vectors):
        transformed = torch.matmul(time_vectors, self.fourier_weights)
        transformed = transformed + self.fourier_bias[None, None, :]

        cos_component = torch.cos(transformed)
        sin_component = torch.sin(transformed)

        mask_zero_timestemp = torch.all(time_vectors != -1, dim = -1, keepdim = True)
        fourier_features = torch.cat([cos_component, sin_component], dim=-1) * mask_zero_timestemp
        assert torch.all(fourier_features[:,0,:] == 0)
        return fourier_features
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        time_vectors=None,
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
        if time_vectors is not None:
            time_embeddings = self._apply_random_fourier_transform(time_vectors)

        if inputs_embeds is None and input_ids is not None:
             inputs_embeds = self.inner_model.get_input_embeddings()(input_ids)

        if time_vectors is not None:
            modified_inputs_embeds = inputs_embeds + self.time_embedding_scale * time_embeddings
        else:
            modified_inputs_embeds = inputs_embeds
        
        outputs = self.inner_model(
            input_ids=None, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds = modified_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        return outputs