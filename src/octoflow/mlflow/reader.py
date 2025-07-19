from __future__ import annotations

from collections.abc import Generator
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
