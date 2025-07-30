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
from typing import TypedDict

import yaml
from tqdm import auto as tqdm

from octoflow.models.experiment import (
    Experiment,
    create_experiment,
    get_experiment_by_name,
)
from octoflow.models.run import (
    Run,
    create_run,
    delete_run,
    find_run_by_name_in_experiment,
)


class RunMeta(TypedDict):
    run_name: str
    start_time: float
    end_time: float
    status: int
    report: dict | None


class ExperimentReader:
    def __init__(self, expr_dir: str | Path) -> None:
        if isinstance(expr_dir, str):
            expr_dir = Path(expr_dir)
        if not expr_dir.exists():
            raise ValueError(f"expr_dir '{expr_dir}' does not exist")
        self.expr_dir = expr_dir

    def count_runs(self) -> int:
        raise NotImplementedError

    def iter_runs(self) -> Generator[RunMeta, None, None]:
        raise NotImplementedError


def import_to_db(reader: ExperimentReader) -> None:
    for run_path in tqdm.tqdm(
        list(reader.expr_dir.iterdir()), desc="Syncing experiments"
    ):
        if not run_path.is_dir():
            continue
        # read the meta.yaml file
        meta_file = run_path / "mlruns" / "0" / "meta.yaml"
        if not meta_file.exists():
            continue

        expr_meta: dict = {}
        with open(meta_file, "r") as f:
            expr_meta.update(yaml.safe_load(f))

        expr_meta.pop("name")

        expr = get_experiment_by_name(run_path.name)

        if not expr:
            creation_time = expr_meta["creation_time"]
            expr = Experiment(name=run_path.name, creation_time=creation_time)
            expr = create_experiment(expr)

        total = reader.count_runs()

        pbar = tqdm.tqdm(total=total, desc="Loading runs", leave=False)

        for r in reader.iter_runs():
            if r.get("report") is not None:
                run_name = r["run_name"]
                run_start_time = r["start_time"]
                run_end_time = r["end_time"]
                run_status = r["status"]
                existing_run = find_run_by_name_in_experiment(
                    experiment_id=expr.id, name=run_name
                )
                if existing_run and run_start_time > existing_run.start_time:
                    existing_run = delete_run(r)  # returns None
                if existing_run:
                    run = existing_run
                else:
                    run = Run(
                        experiment_id=expr.id,
                        name=run_name,
                        start_time=run_start_time,
                        end_time=run_end_time,
                        status=run_status,
                        meta=r,
                    )
                    create_run(run)
            pbar.update(1)


# ------------------------------------------
# MLflow wrapper functions
# ------------------------------------------


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


def set_tracking_uri(tracking_uri: str | Path | None = None):
    if not tracking_uri:
        return
    # uri: Union[str, Path]
    mlflow.set_tracking_uri(tracking_uri)


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
