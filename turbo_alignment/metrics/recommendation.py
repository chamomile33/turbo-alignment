import numpy as np
from typing import List, Any, Tuple, Optional, Iterator
import json
from pathlib import Path
import torch

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import RecommendationMetricsSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType

logger = get_project_logger()

def batched(iterable, batch_size: int) -> Iterator[List[Any]]:
    length = len(iterable)
    for i in range(0, length, batch_size):
        yield iterable[i:min(i + batch_size, length)]


@Metric.register(MetricType.RECOMMENDATION_METRIC)
class RecommendationMetric(Metric):
    def __init__(self, settings: RecommendationMetricsSettings, item_generation_output = None) -> None:
        super().__init__(settings=settings)
        self._settings: RecommendationMetricsSettings = settings
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._batch_size = getattr(self._settings, 'batch_size', 128)
        self._max_top_k = max(self._settings.top_k)
        
        if item_generation_output is None:
            self._item_embeddings, self._item_ids = self._load_item_embeddings()
        else:
            self._item_ids, self._item_embeddings = item_generation_output
            self._item_embeddings = torch.tensor(np.array(self._item_embeddings), dtype=torch.float32).to(self._device)

    
    def _load_item_embeddings(self) -> Tuple[torch.Tensor, List[str]]:
        if not self._settings.item_embeddings_path:
            raise ValueError("Путь к эмбеддингам айтемов не указан")
        
        embeddings_path = Path(self._settings.item_embeddings_path)
        
        item_embeddings = []
        item_ids = []
        
        with open(embeddings_path, 'r') as f:
            for line in f:
                item_data = json.loads(line)
                item_ids.append(item_data['item_id'])
                item_embeddings.append(np.array(item_data['embedding']))

        items_tensor = torch.tensor(np.array(item_embeddings), dtype=torch.float32).to(self._device)
        logger.info(f"Загружено {len(item_ids)} эмбеддингов айтемов")
        return items_tensor, item_ids

    def compute(self, **kwargs) -> list[MetricResults]:
        embeddings: List[np.ndarray] = kwargs.get('embeddings', None)
        true_items: List[List[str]] = kwargs.get('true_items', None)
        dataset_name: str = kwargs.get('dataset_name', '')
        
        if embeddings is None or true_items is None:
            raise ValueError('Embeddings или true_items не должны быть None')
        
        precision_scores = {k: [] for k in self._settings.top_k}
        recall_scores = {k: [] for k in self._settings.top_k}
        ndcg_scores = {k: [] for k in self._settings.top_k}
        ap_scores = {k: [] for k in self._settings.top_k}
        
        for batch_start in range(0, len(embeddings), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(embeddings))
            logger.info(f"Обработка батча пользователей {batch_start}:{batch_end} из {len(embeddings)}")
            
            batch_embeddings = embeddings[batch_start:batch_end]
            batch_true_items = true_items[batch_start:batch_end]
            batch_user_embeddings = torch.tensor(np.array(batch_embeddings), dtype=torch.float32).to(self._device)
            top_k_indices, _ = self._get_top_k_items_batch(batch_user_embeddings, k=self._max_top_k)
            
            batch_top_items = [[self._item_ids[idx] for idx in indices] for indices in top_k_indices.cpu().numpy()]
            
            for user_true_items, user_top_items in zip(batch_true_items, batch_top_items):
                for k in self._settings.top_k:
                    user_top_k_items = user_top_items[:k]
                    
                    precision = self._calculate_precision_at_k(user_top_k_items, user_true_items)
                    precision_scores[k].append(precision)
                    
                    recall = self._calculate_recall_at_k(user_top_k_items, user_true_items)
                    recall_scores[k].append(recall)
                    
                    ndcg = self._calculate_ndcg_at_k(user_top_k_items, user_true_items)
                    ndcg_scores[k].append(ndcg)
                    
                    ap = self._calculate_ap_at_k(user_top_k_items, user_true_items)
                    ap_scores[k].append(ap)
        
        results = []
        
        for k in self._settings.top_k:
            for need_average in self._settings.need_average:
                print(f'MEAN PRECISION {k}',np.array(precision_scores[k]).mean(), len(precision_scores[k]))
                results.append(
                    MetricResults(
                        element_wise_scores=[
                            ElementWiseScores(
                                label=f"{dataset_name}@@precision@{k}",
                                values=precision_scores[k],
                            )
                        ],
                        need_average=need_average,
                    )
                )
        
        for k in self._settings.top_k:
            for need_average in self._settings.need_average:
                print(f'MEAN RECALL {k}',np.array(recall_scores[k]).mean())
                results.append(
                    MetricResults(
                        element_wise_scores=[
                            ElementWiseScores(
                                label=f"{dataset_name}@@recall@{k}",
                                values=recall_scores[k],
                            )
                        ],
                        need_average=need_average,
                    )
                )
        
        for k in self._settings.top_k:
            for need_average in self._settings.need_average:
                print(f'MEAN NDGC {k}',np.array(ndcg_scores[k]).mean())
                results.append(
                    MetricResults(
                        element_wise_scores=[
                            ElementWiseScores(
                                label=f"{dataset_name}@@ndcg@{k}",
                                values=ndcg_scores[k],
                            )
                        ],
                        need_average=need_average,
                    )
                )
        
        for k in self._settings.top_k:
            for need_average in self._settings.need_average:
                print(f'MEAN AP {k}',np.array(ap_scores[k]).mean())
                results.append(
                    MetricResults(
                        element_wise_scores=[
                            ElementWiseScores(
                                label=f"{dataset_name}@@ap@{k}",
                                values=ap_scores[k],
                            )
                        ],
                        need_average=need_average,
                    )
                )
        
        return results
    
    def _get_top_k_items_batch(self, user_embeddings: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if k is None:
            k = self._max_top_k
            
        similarities = torch.matmul(user_embeddings, self._item_embeddings.T)
        values, indices = torch.topk(similarities, k=k, dim=1,largest=True, sorted=True)
        return indices, values
    
    def _get_top_k_items(self, user_embedding: np.ndarray, k: Optional[int] = None) -> List[str]:
        if k is None:
            k = self._max_top_k
            
        user_embedding_tensor = torch.tensor(
            user_embedding, dtype=torch.float32
        ).unsqueeze(0).to(self._device)
        indices, _ = self._get_top_k_items_batch(user_embedding_tensor, k=k)
        return [self._item_ids[idx] for idx in indices[0].cpu().numpy()]
    
    def _calculate_precision_at_k(self, recommended_items: List[str], true_items: List[str]) -> float:
        hits = len(set(recommended_items) & set(true_items))
        return hits / len(recommended_items)
    
    def _calculate_recall_at_k(self, recommended_items: List[str], true_items: List[str]) -> float:
        hits = len(set(recommended_items) & set(true_items))
        return hits / len(true_items)
    
    def _calculate_ndcg_at_k(self, recommended_items: List[str], true_items: List[str]) -> float:
        true_set = set(true_items)
        dcg = 0.0
        idcg = 0.0
        
        for i, item_id in enumerate(recommended_items):
            if item_id in true_set:
                dcg += 1.0 / np.log2(i + 2)
        
        for i in range(min(len(true_items), len(recommended_items))):
            idcg += 1.0 / np.log2(i + 2) 
        
        return dcg / idcg
    
    def _calculate_ap_at_k(self, recommended_items: List[str], true_items: List[str]) -> float:
        true_set = set(true_items)
        cum_precision = 0.0
        num_hits = 0
        
        for i, item in enumerate(recommended_items):
            if item in true_set:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                cum_precision += precision_at_i
        
        divisor = min(len(recommended_items), len(true_items))
        return cum_precision / divisor 


@Metric.register(MetricType.RECALL)
class RecallMetric(RecommendationMetric):
    pass

@Metric.register(MetricType.PRECISION)
class PrecisionMetric(RecommendationMetric):
    pass


@Metric.register(MetricType.NDCG)
class NDCGMetric(RecommendationMetric):
    pass 