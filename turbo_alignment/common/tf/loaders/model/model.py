import torch
from peft import PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
import torch.nn as nn

from turbo_alignment.common import is_package_available
from turbo_alignment.common.tf.loaders.model.registry import (
    PeftConfigRegistry,
    TransformersAutoModelRegistry,
)
from turbo_alignment.settings.model import (
    ModelForPeftSettings,
    ModelType,
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
    ModelWithMlpSettings,
    TimeAwareModelSettings,
    CollabEmbedsModelSettings,
    CrossAttentionModelSettings,
    CollabAttentionModelSettings,
    CollabCrossAttentionModelSettings
)

from turbo_alignment.settings.tf.peft import PEFT_TYPE
from turbo_alignment.common.tf.loaders.model.time_aware_model import TimeAwareConfig, TimeAwareModel
from turbo_alignment.common.tf.loaders.model.collab_embeds_model import CollabEmbedsConfig, CollaborativeEmbeddingsModel
from turbo_alignment.common.tf.loaders.model.cross_attention_model import CrossAttentionQwen2Config, CrossAttentionQwen2Model
from turbo_alignment.common.tf.loaders.model.collab_attention_model import CollabAttentionConfig, CollabAttentionModel
from turbo_alignment.common.tf.loaders.model.collab_cross_attention_model import CollabCrossAttentionConfig, CollabCrossAttentionModel

from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
import torch.nn as nn
from dataclasses import dataclass

class ModelWithMlpConfig(PretrainedConfig):
    model_type = "model-with-mlp"

    def __init__(self,
                 mlp_intermediate_size: int = 1024,
                 mlp_initializer_range: float = 0.02,
                 mlp_hidden_act: str = "silu",
                 mlp_pooling_type: str = "mean",
                 base_model_name: str = "",
                 **kwargs):
        super().__init__(**kwargs)
        self.mlp_intermediate_size = mlp_intermediate_size
        self.mlp_initializer_range = mlp_initializer_range
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_pooling_type = mlp_pooling_type
        self.base_model_name = base_model_name
        self._attn_implementation_autoset = False
        self.attn_implementation = "flash_attention_2"
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "ModelWithMlpConfig":
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


class ModelWithMlp(PreTrainedModel):

    config_class = ModelWithMlpConfig
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
    
    def __init__(self, config: ModelWithMlpConfig):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        super().__init__(config)
        config._attn_implementation_autoset = False
        config.attn_implementation = "flash_attention_2"
        config.base_model_name = 'RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4'
        inner_conf = AutoConfig.from_pretrained(config.base_model_name)
        dict_for_update = {key: val for key, val in config.to_dict().items() if key not in ['_name_or_path', 'architectures']}
        inner_conf.update(dict_for_update)
        self.inner_model = AutoModel.from_config(inner_conf)
        config.mlp_hidden_size = self.inner_model.config.hidden_size

        @dataclass
        class QwenCFG:
            hidden_size: int
            intermediate_size: int
            hidden_act: str

        cfg = QwenCFG(hidden_size = config.mlp_hidden_size,
                            intermediate_size = config.mlp_intermediate_size,
                            hidden_act = config.mlp_hidden_act)
        self.mlp = Qwen2MLP(config = cfg)
        self.pooling_type = config.mlp_pooling_type
        self._init_mlp_weights()
        self.post_init()
    
    def _init_mlp_weights(self):
        for layer in [self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj]:
            nn.init.normal_(layer.weight, mean=0.0, std=self.config.mlp_initializer_range)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                **kwargs):
        outputs = self.inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        last_hidden_states = outputs.last_hidden_state
        if self.pooling_type == "mean":
            expanded_mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
            masked_hidden = last_hidden_states * expanded_mask
            sum_mask = expanded_mask.sum(dim=1) + 1e-9
            pooled_output = masked_hidden.sum(dim=1) / sum_mask
        
        elif self.pooling_type == "last":
            seq_lengths = attention_mask.sum(dim=1, dtype=torch.int64) - 1
            batch_size = last_hidden_states.size(0)
            batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
            pooled_output = last_hidden_states[batch_indices, seq_lengths]
        else:
            raise ValueError(f"Неподдерживаемый тип пулинга: {self.pooling_type}")

        mlp_output = self.mlp(pooled_output)
        outputs.last_hidden_state = mlp_output.unsqueeze(1)
        return outputs
    
AutoConfig.register("model-with-mlp", ModelWithMlpConfig)
AutoModel.register(ModelWithMlpConfig, ModelWithMlp)
AutoConfig.register("time-aware-model", TimeAwareConfig)
AutoModel.register(TimeAwareConfig, TimeAwareModel)
AutoConfig.register("collab-embeds-model", CollabEmbedsConfig)
AutoModel.register(CollabEmbedsConfig, CollaborativeEmbeddingsModel)
AutoConfig.register("cross_attention_qwen2", CrossAttentionQwen2Config)
AutoModel.register(CrossAttentionQwen2Config, CrossAttentionQwen2Model)
AutoConfig.register("collab_attention", CollabAttentionConfig)
AutoModel.register(CollabAttentionConfig, CollabAttentionModel)
AutoConfig.register("collab_cross_attention", CollabCrossAttentionConfig)
AutoModel.register(CollabCrossAttentionConfig, CollabCrossAttentionModel)


def _prepare_model_for_peft(model: PreTrainedModel, peft_settings: PEFT_TYPE) -> PeftModel:
    peft_params = peft_settings.dict()
    peft_params.pop('name')

    peft_config = PeftConfigRegistry.by_name(peft_settings.name)(**peft_params)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return get_peft_model(model, peft_config)


def _load_pretrained_adapters(
    model: PreTrainedModel,
    model_settings: PreTrainedAdaptersModelSettings,
) -> PeftModel:
    return PeftModel.from_pretrained(
        model,
        model_settings.adapter_path,
        is_trainable=model_settings.is_trainable,
    )

def unfreeze_params(layer):
    for param in layer.parameters():
        param.requires_grad = True


def load_model(
    model_settings: PreTrainedModelSettings,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:
    # pylint: disable=import-error
 
    if model_settings.liger_kernels_settings is not None and is_package_available('liger-kernel'):
        from liger_kernel.transformers import (
            apply_liger_kernel_to_gemma2,
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_qwen2,
        )

        apply_liger_kernel_to_llama(
            rope=model_settings.liger_kernels_settings.use_rope,
            cross_entropy=model_settings.liger_kernels_settings.use_cross_entropy,
            swiglu=model_settings.liger_kernels_settings.use_mlp,
            rms_norm=model_settings.liger_kernels_settings.use_rms_norm,
            fused_linear_cross_entropy=model_settings.liger_kernels_settings.use_fused_linear_cross_entropy,
        )

        apply_liger_kernel_to_gemma2(
            rope=model_settings.liger_kernels_settings.use_rope,
            cross_entropy=model_settings.liger_kernels_settings.use_cross_entropy,
            geglu=model_settings.liger_kernels_settings.use_mlp,
            rms_norm=model_settings.liger_kernels_settings.use_rms_norm,
            fused_linear_cross_entropy=model_settings.liger_kernels_settings.use_fused_linear_cross_entropy,
        )

        apply_liger_kernel_to_qwen2(
            rope=model_settings.liger_kernels_settings.use_rope,
            cross_entropy=model_settings.liger_kernels_settings.use_cross_entropy,
            swiglu=model_settings.liger_kernels_settings.use_mlp,
            rms_norm=model_settings.liger_kernels_settings.use_rms_norm,
            fused_linear_cross_entropy=model_settings.liger_kernels_settings.use_fused_linear_cross_entropy,
        )

    model = TransformersAutoModelRegistry.by_name(model_settings.model_type).from_pretrained(
        model_settings.model_path,
        **model_settings.transformers_settings.dict(exclude_none=True),
        **model_settings.model_kwargs,
        use_flash_attention_2 = True 
    )

    print('MODEL', model)

    if model_settings.transformers_settings.load_in_8bit:
        model = prepare_model_for_int8_training(model)

    if model_settings.resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if model_settings.embeddings_initialization_strategy is not None:
        with torch.no_grad():
            for new_token, old_token in model_settings.embeddings_initialization_strategy.items():
                new_token_id = tokenizer.get_added_vocab()[new_token]
                old_token_id = tokenizer.encode(old_token, add_special_tokens=False)[0]

                if model.config.model_type == 'gpt_neox':
                    model.gpt_neox.embed_in.weight[new_token_id, :] = torch.clone(
                        model.gpt_neox.embed_in.weight[old_token_id, :]
                    )
                    if model_settings.model_type == 'causal':
                        model.embed_out.weight[new_token_id, :] = torch.clone(model.embed_out.weight[old_token_id, :])

                elif model.config.model_type == 'llama':
                    model.model.embed_tokens.weight[new_token_id, :] = model.model.embed_tokens.weight[old_token_id, :]

    if isinstance(model_settings, PreTrainedAdaptersModelSettings):
        model = _load_pretrained_adapters(model, model_settings)
    elif isinstance(model_settings, ModelForPeftSettings):
        model = _prepare_model_for_peft(model, model_settings.peft_settings)

        if model_settings.model_type == ModelType.SEQ_CLS and is_deepspeed_zero3_enabled():
            model.base_model.model.score = torch.nn.Linear(
                in_features=model.base_model.model.score.original_module.in_features,
                out_features=model.base_model.model.score.original_module.out_features,
                bias=model.base_model.model.score.original_module.bias,
            )
            model.base_model.model.score.weight.requires_grad = True

    elif isinstance(model_settings, ModelWithMlpSettings) and model_settings.add_mlp_layer:
        base_cfg = model.config
        cfg_dict = base_cfg.to_dict()

        mlp_dict = {
                "hidden_size": base_cfg.hidden_size,
                "mlp_intermediate_size": model_settings.mlp_intermediate_size,
                "mlp_initializer_range": model_settings.mlp_initializer_range,
                "mlp_hidden_act": model_settings.mlp_hidden_act,
                "mlp_pooling_type" : model_settings.mlp_pooling_type,
                "model_type": ModelWithMlpConfig.model_type,
                "base_model_name" : base_cfg._name_or_path
        }
        cfg_dict.update(mlp_dict)

        custom_cfg = ModelWithMlpConfig.from_dict(cfg_dict)

        new_model = AutoModel.from_config(custom_cfg)
        new_model.inner_model = model
        if model_settings.freeze_base_model:
            for param in new_model.inner_model.parameters():
                param.requires_grad = False
        for param in new_model.mlp.parameters():
                param.requires_grad = True     
        model = new_model
        model.enable_input_require_grads()
    
    elif isinstance(model_settings, TimeAwareModelSettings) and model_settings.is_time_aware_model:
        base_cfg = model.config
        cfg_dict = base_cfg.to_dict()

        time_dict = {
                "time_dim": model_settings.time_dim,
                "model_type": TimeAwareConfig.model_type,
                "base_model_name" : base_cfg._name_or_path
        }
        cfg_dict.update(time_dict)

        custom_cfg = TimeAwareConfig.from_dict(cfg_dict)

        new_model = AutoModel.from_config(custom_cfg)
        new_model.inner_model = model

        new_model.fourier_weights.requires_grad = True
        new_model.fourier_bias.requires_grad = True
        new_model.time_embedding_scale.requires_grad = True
        
        new_model.init_fourier()
        model = new_model
    
    elif isinstance(model_settings, CollabEmbedsModelSettings) and model_settings.is_collab_embeds_model:
        base_cfg = model.config
        cfg_dict = base_cfg.to_dict()

        params_dict = {
                "embed_dim": model_settings.embed_dim,
                "projector_hidden_dim": model_settings.projector_hidden_dim,
                "model_type": CollabEmbedsConfig.model_type,
                "base_model_name" : base_cfg._name_or_path
        }
        cfg_dict.update(params_dict)

        custom_cfg = CollabEmbedsConfig.from_dict(cfg_dict)

        new_model = AutoModel.from_config(custom_cfg)
        new_model.inner_model = model
        new_model.init_mlp_weights()
       
        for name, param in new_model.projector.named_parameters():
            param.requires_grad = True

        model = new_model
    
    elif isinstance(model_settings, CrossAttentionModelSettings) and model_settings.is_cross_attention_model:
        base_cfg = model.config
        cfg_dict = base_cfg.to_dict()

        params_dict = {
                "collaborative_embedding_dim": model_settings.collaborative_embedding_dim,
                "num_cross_layers": model_settings.num_cross_layers,
                "model_type": CrossAttentionQwen2Config.model_type,
                "base_model_name" : base_cfg._name_or_path
        }
        cfg_dict.update(params_dict)

        custom_cfg = CrossAttentionQwen2Config.from_dict(cfg_dict)

        new_model = AutoModel.from_config(custom_cfg)
        new_model.inner_model = model.model
        new_model.replace_layers_with_cross_attention()

        print('NEW MODEL', new_model, flush = True)
        for name, param in new_model.embedding_projector.named_parameters():
            param.requires_grad = True
        for param in new_model.parameters():
            param.requires_grad = True

        model = new_model
    
    elif isinstance(model_settings, CollabAttentionModelSettings) and model_settings.is_collab_attention_model:
        base_cfg = model.config
        cfg_dict = base_cfg.to_dict()

        params_dict = {
                "collaborative_embedding_dim": model_settings.collaborative_embedding_dim,
                "num_collaborative_layers": model_settings.num_collaborative_layers,
                "projector_hidden_ratio": model_settings.projector_hidden_ratio,
                "attention_fusion_type": model_settings.attention_fusion_type,
                "model_type": CollabAttentionConfig.model_type,
                "base_model_name" : base_cfg._name_or_path
        }
        cfg_dict.update(params_dict)

        custom_cfg = CollabAttentionConfig.from_dict(cfg_dict)

        new_model = AutoModel.from_config(custom_cfg)
        new_model.inner_model = model.model
        new_model.replace_layers_with_collab_attention()

        print('NEW MODEL', new_model, flush = True)
        for name, param in new_model.embedding_projector.named_parameters():
            param.requires_grad = True
        for param in new_model.parameters():
            param.requires_grad = True

        model = new_model
    
    elif isinstance(model_settings, CollabCrossAttentionModelSettings) and model_settings.is_collab_cross_attention_model:
        base_cfg = model.config
        cfg_dict = base_cfg.to_dict()

        params_dict = {
                "collaborative_embedding_dim": model_settings.collaborative_embedding_dim,
                "num_collaborative_layers": model_settings.num_collaborative_layers,
                "projector_hidden_ratio": model_settings.projector_hidden_ratio,
                "attention_fusion_type": model_settings.attention_fusion_type,
                "model_type": CollabCrossAttentionConfig.model_type,
                "base_model_name" : base_cfg._name_or_path
        }

        cfg_dict.update(params_dict)
        
        custom_cfg = CollabCrossAttentionConfig.from_dict(cfg_dict)

        new_model = AutoModel.from_config(custom_cfg)
        new_model.inner_model = model.model
        new_model.replace_layers_with_collab_cross_attention()

        print('NEW MODEL', new_model, flush = True)
        for name, param in new_model.embedding_projector.named_parameters():
            param.requires_grad = True
        for name, param in new_model.collab_projector.named_parameters():
            param.requires_grad = True
        for param in new_model.parameters():
            param.requires_grad = True

        model = new_model

    return model
