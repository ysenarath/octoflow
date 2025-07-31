from __future__ import annotations

from collections.abc import Generator
import traceback
from contextlib import contextmanager
from typing import Any, Optional

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import sklearn as mlflow_sklearn

from pathlib import Path
from packaging import version


if version.parse(mlflow.__version__) >= version.parse("3.0.0"):
    err = f"mlflow version {mlflow.__version__} is not compatible with this code"
    raise ImportError(err)


def set_experiment(experiment_name: str) -> None:
    experiment_id = None
    # Ensure that the experiment is created in the database
    #   name: str
    #   artifact_location: Optional[str] = None
    #   tags: Optional[dict[str, Any]] = None
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location=None,
        tags=None,
    )
    # Set the experiment ID in the current run
    #   experiment_name: Optional[str] = None
    #   experiment_id: Optional[str] = None
    mlflow.set_experiment(experiment_name=experiment_name, experiment_id=experiment_id)


def set_tracking_uri(uri: str | Path | None = None):
    # uri: Union[str, Path]
    if not uri:
        return
    mlflow.set_tracking_uri(uri)


def get_tracking_uri() -> str:
    return mlflow.get_tracking_uri()


@contextmanager
def start_run(
    run_name: Optional[str] = None, nested: bool = False
) -> Generator[mlflow.ActiveRun, None, None]:
    """
    run_id: str | None = None,
    experiment_id: str | None = None,
    run_name: str | None = None,
    nested: bool = False,
    parent_run_id: str | None = None,
    tags: dict[str, Any] | None = None,
    description: str | None = None,
    log_system_metrics: bool | None = None
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield run


def log_param(key: str, value: Any) -> None:
    mlflow.log_param(key, value)


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    mlflow.log_metric(key, value, step=step)


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_figure(figure: plt.Figure, artifact_file: str) -> None:
    mlflow.log_figure(figure, artifact_file)


def log_table(data: pd.DataFrame, artifact_file: str) -> None:
    mlflow.log_table(data, artifact_file=artifact_file)


def log_text(text: str, artifact_file: str) -> None:
    mlflow.log_text(text, artifact_file=artifact_file)


def log_error(artifact_file: str = "logs/error.txt") -> None:
    error_trace = traceback.format_exc()
    log_text(error_trace, artifact_file)


sklearn = mlflow_sklearn
