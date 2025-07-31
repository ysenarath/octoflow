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
    @property
    def metadata(self) -> dict:
        if not hasattr(self, "_metadata"):
            self._metadata = self.read_metadata()
        return self._metadata

    def read_metadata(self) -> dict:
        raise NotImplementedError

    @property
    def name(self) -> str:
        if not hasattr(self, "_name"):
            self.name = self.metadata.get("name", "Default")
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @classmethod
    def list_experiment_readers(cls, *args, **kwargs) -> list[ExperimentReader]:
        raise NotImplementedError

    def count_runs(self) -> int:
        raise NotImplementedError

    def iter_runs(self) -> Generator[RunMeta, None, None]:
        raise NotImplementedError

    @classmethod
    def import_to_db(cls, *args, **kwargs) -> None:
        readers = cls.list_experiment_readers(*args, **kwargs)

        for reader in tqdm.tqdm(readers, desc="Syncing experiments"):
            expr = get_experiment_by_name(reader.name)

            print(f"Processing experiment: {reader.name}")
            print(f"Experiment metadata: {reader.metadata}")

            if not expr:
                creation_time = reader.metadata["creation_time"]
                expr = Experiment(name=reader.name, creation_time=creation_time)
                expr = create_experiment(expr)

            total = reader.count_runs()

            if not total:
                print(f"No runs found in experiment {reader.name}. Skipping.")
                continue

            pbar = tqdm.tqdm(total=total, desc="Loading runs", leave=False)

            for r in reader.iter_runs():
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


class MLflowExperimentReader(ExperimentReader):
    def __init__(self, path: str | Path, name: str, experiment_id: str = "0") -> None:
        super().__init__()
        self.path = Path(path)
        self.name = name
        self.experiment_id = str(experiment_id)

    def read_metadata(self) -> dict:
        meta_file = self.path / "meta.yaml"
        if not meta_file.exists():
            return None
        expr_meta: dict = {}
        with open(meta_file, "r") as f:
            expr_meta.update(yaml.safe_load(f))
        return expr_meta

    def count_runs(self) -> int:
        return len(list(self.path.glob("*")))

    def iter_runs(self) -> Generator[RunMeta, None, None]:
        for run_dir in self.path.glob("*"):
            if not run_dir.is_dir():
                continue
            runs_meta_path = run_dir / "meta.yaml"
            if not runs_meta_path.exists():
                continue
            with open(runs_meta_path, "r") as f:
                meta = RunMeta(**yaml.safe_load(f))
            yield meta

    @classmethod
    def list_experiment_readers(cls, path: str | Path) -> list[MLflowExperimentReader]:
        readers = []
        for expr_meta_path in path.glob("*/mlruns/*/meta.yaml"):
            expr_path = expr_meta_path.parent.resolve().absolute()
            expr_name = str(expr_path.relative_to(path)).replace("/mlruns/", "/")
            reader = cls(
                path=expr_path,
                experiment_id=expr_meta_path.parent.name,
                name=expr_name,
            )
            readers.append(reader)
        return readers
