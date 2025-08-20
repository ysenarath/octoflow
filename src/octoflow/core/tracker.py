from __future__ import annotations

from collections.abc import Generator

from tqdm import auto as tqdm

from octoflow.models.experiment import (
    Experiment,
    create_experiment,
    get_experiment_by_name,
)
from octoflow.models.run import Run, RunMeta, create_run, delete_run, find_run_by_name


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

        for reader in tqdm.tqdm(readers, desc="Importing experiments"):
            expr = get_experiment_by_name(reader.name)

            if not expr:
                creation_time = reader.metadata["creation_time"]
                expr = Experiment(name=reader.name, creation_time=creation_time)
                expr = create_experiment(expr)

            total = reader.count_runs()

            if not total:
                continue

            print(f"Found {total} runs in experiment '{expr.name}'")

            pbar = tqdm.tqdm(total=total, desc="Loading runs")

            for data in reader.iter_runs():
                run_name = data["run_name"]
                run_start_time = data["start_time"]
                run_end_time = data["end_time"]
                run_status = data["status"]
                existing_run = find_run_by_name(experiment_id=expr.id, name=run_name)
                if existing_run and run_start_time > existing_run.start_time:
                    existing_run = delete_run(existing_run)
                if existing_run:
                    run = existing_run
                else:
                    run = Run(
                        experiment_id=expr.id,
                        name=run_name,
                        start_time=run_start_time,
                        end_time=run_end_time,
                        status=run_status,
                        data=data,
                    )
                    create_run(run)
                pbar.update(1)
