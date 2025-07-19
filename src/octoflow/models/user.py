from __future__ import annotations

import enum
import uuid
from typing import Optional, Union

from flask_login import LoginManager, current_user
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column
from werkzeug.security import check_password_hash, generate_password_hash

from octoflow.models.base import db

__all__ = [
    "login_manager",
]

login_manager = LoginManager()

current_user: Optional[User]


class UserRole(enum.Enum):
    USER = "user"
    ADMIN = "admin"
    ANONYMOUS = "anonymous"


class UserStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"


def create_alt_id() -> str:
    """Generate a unique alt_id for the user."""
    return str(uuid.uuid4())


class User(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    password_hash: Mapped[Optional[str]] = mapped_column(nullable=True)
    email: Mapped[Optional[str]] = mapped_column(
        unique=True, nullable=True, default=None
    )
    role: Mapped[UserRole] = mapped_column(default=UserRole.USER, nullable=False)
    status: Mapped[UserStatus] = mapped_column(
        default=UserStatus.ACTIVE, nullable=False
    )
    alternative_id: Mapped[str] = mapped_column(
        unique=True, nullable=False, default=None
    )

    @property
    def password(self):
        raise AttributeError("password is not a readable attribute")

    @password.setter
    def password(self, password: str):
        self.password_hash = generate_password_hash(password)
        self.alternative_id = create_alt_id()

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return self.status == UserStatus.ACTIVE

    @property
    def is_anonymous(self):
        return self.role == UserRole.ANONYMOUS

    def get_id(self):
        return self.alternative_id

    def check_password(self, pwd: str) -> bool:
        return check_password_hash(self.password_hash, pwd)


def list_users() -> list[User]:
    return User.query.all()


def get_user_by_username(username: str) -> Optional[User]:
    user = User.query.filter_by(username=username).first()
    if not isinstance(user, User):
        return None
    return user


def update_user(user: User) -> None:
    """Update the user in the database."""
    try:
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        raise ValueError("error occurred while updating the user") from e


def create_user(user: User) -> User:
    try:
        db.session.add(user)
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        raise ValueError("error occurred while creating the user") from e
    return user


def delete_user(usr: Union[User, str, None], remove: bool = False) -> None:
    # validate if self can change the status
    if usr is None:
        usr = current_user.username
    elif isinstance(usr, User):
        usr = usr.username
    if current_user.username != usr and current_user.role != UserRole.ADMIN:
        raise ValueError("only the owner or admin can delete a user")
    user = User.query.filter_by(username=usr).first()
    if not isinstance(user, User):
        raise ValueError("user not found")
    # check if user can be deleted
    if not remove and user.status == UserStatus.DELETED:
        raise ValueError("user already deleted")
    user.status = UserStatus.DELETED
    try:
        if remove:
            db.session.delete(user)
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        raise ValueError("error occurred while deleting the user") from e


@login_manager.user_loader
def _get_user_by_alt_id(user_id: str) -> Optional[User]:
    """Load a user by their unique identifier."""
    try:
        user = User.query.filter_by(alternative_id=user_id).first()
    except SQLAlchemyError:
        return None
    if not isinstance(user, User):
        return None
    return user


def init_default_users() -> None:
    """Initialize the default users."""
    try:
        if not User.query.filter_by(username="admin").first():
            admin = User(
                username="admin",
                role=UserRole.ADMIN,
                status=UserStatus.ACTIVE,
            )
            admin.password = "admin"
            db.session.add(admin)
            db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        raise ValueError(f"error occurred while creating the default users: {e}") from e
