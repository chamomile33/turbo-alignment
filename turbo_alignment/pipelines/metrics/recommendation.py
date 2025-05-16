import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.metrics.recommendation import RecommendationMetric
from turbo_alignment.metrics.registry import RecommendationMetricsSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings, 
    PreTrainedModelSettings,
)
from turbo_alignment.settings.generators.embeddings import EmbeddingsGenerationSettings
from turbo_alignment.generators.embeddings import EmbeddingsGenerator
from turbo_alignment.dataset.recommendation.models import UserRecommendationRecord, TimeAwareUserRecommendationRecord, CollabEmbedsUserRecommendationRecord
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings
from turbo_alignment.settings.datasets.recommendation import RecommendationMultiDatasetSettings
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.trainers.utils import prepare_model_for_deepspeed

logger = get_project_logger()


class RecommendationMetricsStrategy:
    def run(
        self,
        dataset_settings: RecommendationMultiDatasetSettings,
        model_settings: PreTrainedAdaptersModelSettings | PreTrainedModelSettings,
        tokenizer_settings: TokenizerSettings,
        item_embeddings_path: Path,
        output_path: Path,
        top_k: List[int] = [10],
        batch_size: int = 128,
        pooling_strategy: str = 'mean',
        use_accelerator: bool = True,
        deepspeed_config: Optional[Union[Path, Dict[str, Any]]] = None,
        fsdp_config: Optional[Dict[str, Any]] = None,
        max_tokens_count: int = 5000,
    ) -> Dict[str, Any]:
      
        dataset_source = dataset_settings.sources[0]
        dataset_path = Path(getattr(dataset_source, 'path', None) or getattr(dataset_source, 'records_path', None))
        item_embeddings_path = Path(item_embeddings_path)
        output_path = Path(output_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Файл с датасетом не найден: {dataset_path}")
        
        if not item_embeddings_path.exists():
            raise FileNotFoundError(f"Файл с эмбеддингами айтемов не найден: {item_embeddings_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        accelerator = None
        if use_accelerator:
            if deepspeed_config:
                if isinstance(deepspeed_config, Path):
                    ds_plugin = DeepSpeedPlugin(hf_ds_config=str(deepspeed_config))
                else:
                    ds_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config)
                    
                accelerator = Accelerator(deepspeed_plugin=ds_plugin)
                accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = 1 * accelerator.num_processes if type(ds_plugin.deepspeed_config['train_batch_size']) == str else ds_plugin.deepspeed_config['train_batch_size']
            elif fsdp_config:
                accelerator = Accelerator(fsdp_plugin=fsdp_config)
            else:
                accelerator = Accelerator()

        logger.info(f"Загрузка модели и токенизатора")
        tokenizer = load_tokenizer(tokenizer_settings, model_settings)
        model = load_model(model_settings, tokenizer)
        
        if accelerator:
            if deepspeed_config:
                if not hasattr(accelerator.state, 'deepspeed_plugin'):
                    model = model.eval()
                else:
                    model = prepare_model_for_deepspeed(model, accelerator)
            else:
                model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)
        else:
            if torch.cuda.is_available():
                model = model.to("cuda")
            else:
                model = model.to("cpu")
        
        logger.info(f"Загрузка датасета пользователей")
    
        if hasattr(dataset_settings, 'max_tokens_count'):
            dataset_settings.max_tokens_count = max_tokens_count
        
        dataset_loader = DatasetLoader()
        datasets = dataset_loader.load_datasets(
            multi_dataset_settings=dataset_settings,
            tokenizer=tokenizer,
            strategy=DatasetStrategy.INFERENCE,
            seed=42
        )
        
        if not datasets:
            raise ValueError("Не удалось загрузить датасет пользователей")
            
        logger.info(f"Загружено {len(datasets)} датасетов пользователей")
        
        all_user_embeddings = []
        all_true_items = []
        all_user_ids = []
        
        for dataset_idx, dataset in enumerate(datasets):
            logger.info(f"Обработка датасета #{dataset_idx+1}: {dataset.source.name} ({len(dataset)} записей)")
            
            embeddings_settings = EmbeddingsGenerationSettings(
                pooling_strategy=pooling_strategy
            )
            
            if dataset_settings.dataset_type == 'collab_cross_attention':
                record_cls = CollabEmbedsUserRecommendationRecord
            elif dataset_settings.dataset_type == 'time_aware_recommendation':
                record_cls = TimeAwareUserRecommendationRecord
            elif dataset_settings.dataset_type =='collab_embeds' or dataset_settings.dataset_type == 'collab_attention':
                record_cls = CollabEmbedsUserRecommendationRecord
            else:
                record_cls = UserRecommendationRecord

            generator = EmbeddingsGenerator[record_cls](
                model=model,
                tokenizer=tokenizer,
                custom_generation_settings=embeddings_settings,
                batch=batch_size,
                accelerator=accelerator
            )
            
            logger.info(f"Генерация эмбеддингов для датасета {dataset.source.name}")
            generations = generator.generate_from_dataset(dataset)
            
            user_embeddings = [gen.embedding for gen in generations]
            true_items = [gen.item_ids for gen in generations]
            user_ids = [gen.id for gen in generations]
            
            logger.info(f"Сгенерировано {len(user_embeddings)} эмбеддингов пользователей из датасета {dataset.source.name}")
            
            all_user_embeddings.extend(user_embeddings)
            all_true_items.extend(true_items)
            all_user_ids.extend(user_ids)
        
        logger.info(f"Итого сгенерировано {len(all_user_embeddings)} эмбеддингов пользователей из всех датасетов")
       
        metric_settings = RecommendationMetricsSettings(
            top_k=top_k,
            item_embeddings_path=str(item_embeddings_path),
            batch_size=batch_size,
            need_average=[True]
        )
        
        logger.info(f"Вычисление метрик рекомендаций с top_k={top_k}, batch_size={batch_size}")
        
        precision_metric = RecommendationMetric(settings=metric_settings)
        
        results = precision_metric.compute(
            embeddings=all_user_embeddings,
            true_items=all_true_items,
            dataset_name='recommendation'
        )
        
        metrics_data = {}

        for result in results:
            for element_wise_score in result.element_wise_scores:
                label = element_wise_score.label
                if '@@' in label:
                    _, metric_name = label.split('@@', 1)
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = {
                            'user_metrics': {},
                            'average': 0.0
                        }
                    
                    for user_id, value in zip(all_user_ids, element_wise_score.values):
                        metrics_data[metric_name]['user_metrics'][user_id] = value
                    
                    print(f'METRIC {metric_name}, num elements {len(element_wise_score.values)}, type {type(element_wise_score.values)}')
                    metrics_data[metric_name]['average'] = sum(element_wise_score.values) / len(element_wise_score.values)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Результаты сохранены в {output_path}")
    
        for metric_name, data in metrics_data.items():
            logger.info(f"{metric_name}: {data['average']:.4f}")
        
        return metrics_data
    
    @staticmethod
    def _gather_results(data: List[Any], accelerator: Accelerator) -> List[Any]:
        gathered_data = accelerator.gather_for_metrics(data)
        return gathered_data