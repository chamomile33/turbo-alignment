from typing import Any, Dict, List, Optional
from turbo_alignment.dataset.base.models import DatasetRecord


class UserInteraction(DatasetRecord):
    item_id: str
    interaction_type: Optional[str] = None
    timestamp: Optional[int] = None
    additional_data: Optional[Dict[str, Any]] = None

class UserRecommendationRecord(DatasetRecord):
    events_text: str  
    item_ids: List[str] 
    meta: Optional[Dict[str, Any]] = None

class TimeAwareUserRecommendationRecord(UserRecommendationRecord):
    events_timestamps: List[int]

class CollabEmbedsUserRecommendationRecord(UserRecommendationRecord):
    events_embeddings_keys: List[Dict[str, str]]
 
class RecommendationDatasetRecord(DatasetRecord):
    events_text: str  
    items_text: List[str] 
    item_ids: List[int] 
    meta: Optional[Dict[str, Any]] = None

class TimeAwareRecommendationDatasetRecord(RecommendationDatasetRecord):
    events_timestamps: List[int]

class CollabEmbedsDatasetRecord(RecommendationDatasetRecord):
    events_embeddings_keys: List[Dict[str, str]]
    items_embeddings_keys: List[List[Dict[str, str]]]

class ItemRecord(DatasetRecord):
    item_id: str
    item_text: str 
    meta: Optional[Dict[str, Any]] = None

class CollabEmbedsItemRecord(ItemRecord):
    item_embeddings_keys: List[Dict[str, str]]



