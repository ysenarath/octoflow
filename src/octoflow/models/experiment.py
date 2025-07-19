from __future__ import annotations

from typing import Optional

from flask_sqlalchemy.query import Pagination
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from octoflow.models.base import db
from octoflow.models.run import Run


class LifecycleStage:
    """Enum for lifecycle stages of an experiment."""

    ACTIVE = "active"
    DELETED = "deleted"


class Experiment(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    creation_time: Mapped[float] = mapped_column(nullable=False)
    lifecycle_stage: Mapped[str] = mapped_column(
        nullable=False, default=LifecycleStage.ACTIVE
    )
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    runs: Mapped[list[Run]] = relationship(
        "Run", back_populates="experiment", cascade="all, delete-orphan", uselist=True
    )


def list_experiments(
    q: Optional[str] = None,
    order_by: Optional[str] = None,
    reverse: Optional[bool] = None,  # if True returns the largest `order_by` first
    page: int = None,
    per_page: int = 10,
) -> Pagination:
    """List all experiments in the database."""
    query = Experiment.query
    if isinstance(q, str):
        q = q.strip().lower()
        if q:
            query = query.filter(Experiment.name.ilike(f"%{q}%"))
    if order_by:
        order_by = getattr(Experiment, order_by)
        if reverse:
            order_by = order_by.desc()
        query = query.order_by(order_by)
    else:
        order_by = Experiment.creation_time.desc()
        query = query.order_by(order_by)
    return query.paginate(
        page=page,
        per_page=per_page,
        max_per_page=100,
    )


def create_experiment(expr: Experiment) -> Experiment:
    """Add a new experiment to the database."""
    try:
        db.session.add(expr)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise ValueError(f"failed to add experiment: {e}") from e
    return expr


def get_experiment_by_name(name: str) -> Experiment | None:
    """List all experiments by name."""
    return db.session.query(Experiment).filter(Experiment.name == name).first()
