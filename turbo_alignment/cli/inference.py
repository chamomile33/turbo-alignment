from pathlib import Path
from typing import Optional

import typer

from turbo_alignment import pipelines
from turbo_alignment.cli.app import app
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
)
from turbo_alignment.settings.pipelines.inference.chat import (
    ChatInferenceExperimentSettings,
)
from turbo_alignment.settings.pipelines.inference.recommendation import (
    RecommendationInferenceExperimentSettings,
)
from turbo_alignment.settings.pipelines.metrics.recommendation import (
    RecommendationMetricsExperimentSettings,
)
from turbo_alignment.pipelines.metrics.tmp_recommendation import BaseEvalStrategy 
from turbo_alignment.settings import pipelines as pipeline_settings


@app.command(name='inference_chat', help='Infer model on chat dataset')
def infer_chat_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = ChatInferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.ChatInferenceStrategy().run(inference_settings)


@app.command(name='inference_rm', help='Infer model on rm dataset')
def infer_rm_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = InferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.RMInferenceStrategy().run(inference_settings)


@app.command(name='inference_classification', help='Infer model on classification dataset')
def infer_classification_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = InferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.ClassificationInferenceStrategy().run(inference_settings)


@app.command(name='generate_item_embeddings', help='Generate embeddings for items')
def generate_item_embeddings_entrypoint(
    settings_path: Path = typer.Option(
        ..., '--settings_path', exists=True, help='Path to item embeddings generation settings file'
    )
) -> None:

    inference_settings = RecommendationInferenceExperimentSettings.parse_file(settings_path)
    pipelines.ItemEmbeddingsInferenceStrategy().run(inference_settings)


@app.command(name='calculate_recommendation_metrics', help='Calculate recommendation metrics with embedding generation')
def calculate_recommendation_metrics_entrypoint(
    settings_path: Path = typer.Option(
        ..., '--settings_path', exists=True, help='Path to recommendation metrics settings file'
    )
) -> None:

    metrics_settings = RecommendationMetricsExperimentSettings.parse_file(settings_path)
    metrics_strategy = pipelines.RecommendationMetricsStrategy()
    metrics_results = metrics_strategy.run(
        dataset_settings=metrics_settings.dataset_settings,
        model_settings=metrics_settings.model_settings,
        tokenizer_settings=metrics_settings.tokenizer_settings,
        item_embeddings_path=metrics_settings.item_embeddings_path,
        output_path=metrics_settings.output_path,
        top_k=metrics_settings.top_k,
        batch_size=metrics_settings.batch_size,
        pooling_strategy=metrics_settings.pooling_strategy,
        use_accelerator=metrics_settings.use_accelerator,
        deepspeed_config=metrics_settings.deepspeed_config,
        fsdp_config=metrics_settings.fsdp_config,
        max_tokens_count=metrics_settings.max_tokens_count
    )


@app.command(name='calculate_recommendation_metrics_tmp', help='Calculate recommendation metrics with embedding generation')
def calculate_recommendation_metrics_tmp(
    settings_path: Path = typer.Option(
        ..., '--settings_path', exists=True, help='Path to recommendation metrics settings file'
    )
) -> None:
    experiment_settings = pipeline_settings.RecommendationTrainExperimentSettings.parse_file(settings_path)
    BaseEvalStrategy().run(experiment_settings)
