from enum import Enum
from pathlib import Path

from pydantic import model_validator

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.tf.model import ModelTransformersSettings
from turbo_alignment.settings.tf.peft import PEFT_TYPE


class ModelType(str, Enum):
    CAUSAL = 'causal'
    SEQ2SEQ = 'seq2seq'
    SEQ_CLS = 'seq_cls'
    ENC = 'encoder'


class LigerKernelSettings(ExtraFieldsNotAllowedBaseModel):
    use_rope: bool = False
    use_cross_entropy: bool = False
    use_fused_linear_cross_entropy: bool = False
    use_mlp: bool = False
    use_rms_norm: bool = False

    @model_validator(mode='after')
    def check_cross_entopy_kernels(self) -> 'LigerKernelSettings':
        if self.use_fused_linear_cross_entropy and self.use_cross_entropy:
            raise ValueError(
                'You cannot use both FusedLinearCrossEntropy and CrossEntropy kernels. '
                'FusedLinearCrossEntropy is preferred if possible.'
            )

        return self


class PreTrainedModelSettings(ExtraFieldsNotAllowedBaseModel):
    model_path: Path
    model_type: ModelType

    model_kwargs: dict = {}

    transformers_settings: ModelTransformersSettings

    resize_token_embeddings: bool = False

    embeddings_initialization_strategy: dict[str, str] | None = None

    liger_kernels_settings: LigerKernelSettings | None = None


class PreTrainedAdaptersModelSettings(PreTrainedModelSettings):
    adapter_path: Path
    is_trainable: bool = False


class ModelForPeftSettings(PreTrainedModelSettings):
    peft_settings: PEFT_TYPE

class ModelWithMlpSettings(PreTrainedModelSettings):
    add_mlp_layer: bool
    mlp_intermediate_size: int = 8192 
    mlp_initializer_range: float = 0.02
    mlp_hidden_act: str = 'silu'
    freeze_base_model: bool = True
    mlp_pooling_type: str = "mean" 

class TimeAwareModelSettings(PreTrainedModelSettings):
    is_time_aware_model: bool
    time_dim: int = 6 

class CollabEmbedsModelSettings(PreTrainedModelSettings):
    is_collab_embeds_model: bool
    projector_hidden_dim: int = 1024
    embed_dim: int = 300
    
class CrossAttentionModelSettings(PreTrainedModelSettings):
    is_cross_attention_model: bool
    num_cross_layers: int = 3
    collaborative_embedding_dim: int = 300

class CollabAttentionModelSettings(PreTrainedModelSettings):
    is_collab_attention_model: bool
    num_collaborative_layers: int = 3
    collaborative_embedding_dim: int = 300
    projector_hidden_ratio: float = 4
    attention_fusion_type: str = "sum"

class CollabCrossAttentionModelSettings(PreTrainedModelSettings):
    is_collab_cross_attention_model: bool
    num_collaborative_layers: int = 3
    collaborative_embedding_dim: int = 300
    projector_hidden_ratio: float = 4
    attention_fusion_type: str = "sum"