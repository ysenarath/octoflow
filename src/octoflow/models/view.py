from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from flask_login import current_user
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from octoflow.models.base import db
from octoflow.models.run import Run
from octoflow.models.user import User


def timestamp() -> float:
    """Return the current UTC time as a float."""
    return datetime.now(timezone.utc).timestamp()


class AssignedRuns(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"), nullable=False)
    view_id: Mapped[int] = mapped_column(ForeignKey("view.id"), nullable=False)

    run: Mapped[Run] = relationship("Run")
    view: Mapped[View] = relationship("View", back_populates="runs")


class View(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)
    created_at: Mapped[float] = mapped_column(default=timestamp, nullable=False)

    owner: Mapped[User] = relationship("User", backref="views")

    runs: Mapped[List[AssignedRuns]] = relationship(
        "AssignedRuns", back_populates="view", cascade="all, delete-orphan"
    )


def create_view(name: str, owner: User | None = None) -> View:
    """Create a new view and add it to the database."""
    if owner is None:
        owner = current_user
    view = View(name=name, owner=owner)
    try:
        db.session.add(view)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise ValueError("failed to add view") from e
    return view


def get_views(q: Optional[str] = None, user: User | None = None) -> List[View]:
    """Get views from the database."""
    if user is None:
        user = current_user
    if q:
        views = (
            db.session.query(View)
            .filter(View.name.ilike(f"%{q}%"))
            .filter(View.owner_id == user.id)
            .all()
        )
    else:
        views = (
            db.session.query(View)
            .filter(View.owner_id == user.id)
            .order_by(View.created_at.desc())
            .limit(10)
            .all()
        )
    return views


def get_view_by_id(view_id: int | View) -> View:
    """Get a view by its ID."""
    if isinstance(view_id, View):
        view_id = view_id.id
    view = db.session.query(View).filter(View.id == view_id).first()
    if view is None:
        raise ValueError(f"view with id {view_id} not found")
    return view


def add_run_to_view(view_id: int | View, run_id: int | Run) -> AssignedRuns:
    """Add a run to a view."""
    if isinstance(view_id, View):
        view_id = view_id.id
    if isinstance(run_id, Run):
        run_id = run_id.id
    assigned_run = AssignedRuns(run_id=run_id, view_id=view_id)
    try:
        db.session.add(assigned_run)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise ValueError("failed to add run to view") from e
    return assigned_run


def add_runs_to_view(
    view_id: int | View, run_ids: List[int | Run]
) -> List[AssignedRuns]:
    """Add multiple runs to a view."""
    if isinstance(view_id, View):
        view_id = view_id.id
    assigned_runs = []
    for run_id in run_ids:
        if isinstance(run_id, Run):
            run_id = run_id.id
        assigned_run = AssignedRuns(run_id=run_id, view_id=view_id)
        assigned_runs.append(assigned_run)
    try:
        db.session.add_all(assigned_runs)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise ValueError("failed to add runs to view") from e
    return assigned_runs


def remove_run_from_view(view_id: int | View, run_id: int | Run) -> None:
    """Remove a run from a view."""
    if isinstance(view_id, View):
        view_id = view_id.id
    if isinstance(run_id, Run):
        run_id = run_id.id
    assigned_run = (
        db.session.query(AssignedRuns)
        .filter(AssignedRuns.view_id == view_id)
        .filter(AssignedRuns.run_id == run_id)
        .first()
    )
    if assigned_run:
        db.session.delete(assigned_run)
        db.session.commit()
    else:
        msg = f"run with id {run_id} is not assigned to view with id {view_id}"
        raise ValueError(msg)
    return None


def remove_all_runs_from_view(view_id: int | View) -> None:
    """Delete all runs from a view."""
    if isinstance(view_id, View):
        view_id = view_id.id
    assigned_runs = (
        db.session.query(AssignedRuns).filter(AssignedRuns.view_id == view_id).all()
    )
    for assigned_run in assigned_runs:
        db.session.delete(assigned_run)
    db.session.commit()


def delete_view(view_id: int | View) -> None:
    """Delete a view."""
    if isinstance(view_id, View):
        view_id = view_id.id
    view = db.session.query(View).filter(View.id == view_id).first()
    if view:
        db.session.delete(view)
        db.session.commit()
    else:
        msg = f"view with id {view_id} not found"
        raise ValueError(msg)
    return None
