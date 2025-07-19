from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from flask_sqlalchemy.query import Pagination
from sqlalchemy import JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from octoflow.models.base import db

if TYPE_CHECKING:
    from octoflow.models.experiment import Experiment


class RunStatus:
    RUNNING = 1
    SCHEDULED = 2
    FINISHED = 3
    FAILED = 4
    KILLED = 5


class Run(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiment.id"), nullable=False
    )
    start_time: Mapped[float] = mapped_column(nullable=False)
    end_time: Mapped[float] = mapped_column(nullable=True)
    status: Mapped[int] = mapped_column(nullable=False, default=RunStatus.RUNNING)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="runs")

    # create unique constraint for (experiment_id, name)
    __table_args__ = (
        UniqueConstraint("experiment_id", "name", name="uq_experiment_name"),
    )


def create_run(run: Run) -> Run:
    """Add a new run to the database."""
    try:
        db.session.add(run)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise ValueError(f"failed to add run {e}") from e
    return run


def delete_run(run: Run) -> None:
    """Delete a run from the database."""
    try:
        db.session.delete(run)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise ValueError(f"failed to delete run {e}") from e
    return None


def find_run_by_name_in_experiment(experiment_id: int, name: str) -> Run | None:
    """Find a run by name in an experiment."""
    return (
        db.session.query(Run)
        .filter(Run.experiment_id == experiment_id, Run.name == name)
        .first()
    )


def list_runs(
    q: Optional[str] = None,
    order_by: Optional[str] = None,
    reverse: Optional[bool] = None,  # if True returns the largest `order_by` first
    page: int = None,
    per_page: int = 10,
    experiment: Union[Experiment, int, None] = None,
) -> Pagination:
    """List all experiments in the database."""
    query = Run.query
    if isinstance(q, str):
        q = q.strip().lower()
        if q:
            query = query.filter(Run.name.ilike(f"%{q}%"))
    if experiment is not None:
        if not isinstance(experiment, int):
            experiment = experiment.id
        query = query.filter(Run.experiment_id == experiment)
    if order_by:
        order_by = getattr(Run, order_by)
        if reverse:
            order_by = order_by.desc()
        query = query.order_by(order_by)
    else:
        order_by = Run.start_time.desc()
        query = query.order_by(order_by)
    return query.paginate(
        page=page,
        per_page=per_page,
        max_per_page=100,
    )


def get_run_by_id(run_id: int) -> Run | None:
    """Get a run by its ID."""
    return db.session.query(Run).filter(Run.id == run_id).first()
