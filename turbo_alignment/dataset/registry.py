from turbo_alignment.common.registry import Registrable
from turbo_alignment.settings.datasets import DatasetType


class DatasetRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.PAIR_PREFERENCES)
class PairPreferenceDatasetTypeRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.CHAT)
class ChatDatasetTypeRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.CLASSIFICATION)
class ClassificationDatasetTypeRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.SAMPLING)
class SamplingRMDatasetTypeRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.RECOMMENDATION)
class RecommendationDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.TIME_AWARE_RECOMMENDATION)
class TimeAwareRecommendationDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.EMBEDDINGS)
class EmbeddingsDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.COLLAB_EMBEDS)
class CollabEmbedsDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.COLLAB_EMBEDS_EMBEDDINGS)
class CollabEmbedsEmbeddingsDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.CROSS_ATTENTION)
class CrossAttentionDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.COLLAB_ATTENTION)
class CollabAttentionDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.COLLAB_ATTENTION_EMBEDDINGS)
class CollaborativeAttentionEmbeddingsDatasetTypeRegistry(Registrable):
    ...

@DatasetRegistry.register(DatasetType.COLLAB_CROSS_ATTENTION)
class CollabCrossAttentionDatasetTypeRegistry(Registrable):
    ...
