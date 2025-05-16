from typing import Generator, TypeVar, Type

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.generators.embeddings import EmbeddingsGenerator
from turbo_alignment.pipelines.inference.base import BaseInferenceStrategy
from turbo_alignment.settings.pipelines.inference.recommendation import (
    RecommendationInferenceExperimentSettings,
)
from turbo_alignment.dataset.recommendation.models import UserRecommendationRecord, ItemRecord
from turbo_alignment.trainers.utils import prepare_model_for_deepspeed


T = TypeVar('T', UserRecommendationRecord, ItemRecord)


class RecommendationInferenceStrategy(BaseInferenceStrategy[RecommendationInferenceExperimentSettings]):
    def __init__(self, record_type: Type[T] = UserRecommendationRecord):
        super().__init__()
        self.record_type = record_type
    
    def _get_single_inference_settings(
        self, experiment_settings: RecommendationInferenceExperimentSettings, accelerator: Accelerator
    ) -> Generator[tuple[PreTrainedTokenizerBase, BaseGenerator, str, dict], None, None]:
        save_file_id = 0

        for model_inference_settings in experiment_settings.inference_settings:
            tokenizer = load_tokenizer(
                model_inference_settings.tokenizer_settings,
                model_inference_settings.model_settings,
            )

            model = load_model(model_inference_settings.model_settings, tokenizer)
        
            if hasattr(experiment_settings, 'deepspeed_config') and experiment_settings.deepspeed_config:
                if hasattr(accelerator.state, 'deepspeed_plugin'):
                    model = prepare_model_for_deepspeed(model, accelerator)
                else:
                    model = model.eval()
            else:
                model = (
                    accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)
                    if torch.cuda.is_available()
                    else model.to('cpu')  
                )

            for generation_settings in model_inference_settings.generation_settings:
                generator_kwargs = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'custom_generation_settings': generation_settings.custom_settings,
                    'batch': model_inference_settings.batch,
                }
                
                generator = EmbeddingsGenerator[self.record_type](
                    **generator_kwargs,
                    accelerator=accelerator,
                )

                parameters_to_save = {
                    'model_settings': model_inference_settings.model_settings.dict(),
                    'generation_settings': generation_settings.dict(),
                }

                save_file_id += 1
                
                prefix = "user" if self.record_type == UserRecommendationRecord else "item"
                file_name = f'{prefix}_embeddings_inference_{save_file_id}.jsonl'

                yield tokenizer, generator, file_name, parameters_to_save


class UserEmbeddingsInferenceStrategy(RecommendationInferenceStrategy):
    def __init__(self):
        super().__init__(record_type=UserRecommendationRecord)


class ItemEmbeddingsInferenceStrategy(RecommendationInferenceStrategy):
    def __init__(self):
        super().__init__(record_type=ItemRecord) 