from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

from typing_extensions import ParamSpec

from octoflow.data.base import BaseDatasetLoader
from octoflow.data.dataset import DEFAULT_FORMAT, Dataset

loaders: Dict[str, DatasetLoader] = {}

P = ParamSpec("P")
R = TypeVar("R")


class DatasetLoader(BaseDatasetLoader[P, R]):
    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        extensions: Optional[list[str]] = None,
        path_arg: Optional[str] = None,
        wraps: Optional[Callable[P, R]] = None,
    ):
        """
        Initialize a dataset loader.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to decorate.
        name : Optional[str], optional
            The name of the loader, by default None.
        extensions : Optional[list[str]], optional
            The extensions that the loader supports, by default None.
        path_arg : Optional[str], optional
            The name of the argument that is the path, by default None.
        wraps : Optional[Callable[..., Any]], optional
            The function to wrap, by default None.
        """
        super().__init__(func)
        if extensions is None:
            extensions = []
        elif isinstance(extensions, str):
            extensions = [extensions]
        self.name = name or self.func.__name__
        self.extensions = extensions
        self.path_arg = path_arg
        self.wraps = wraps


@overload
def dataloader(
    func: Callable[..., Any],
    name: Optional[str] = None,
    extensions: Optional[list[str]] = None,
    wraps: Optional[Callable[P, R]] = None,
    path_arg: Optional[str] = None,
) -> DatasetLoader[P, R]:
    """
    Decorator to register a function as a dataset loader.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to decorate.
    name : Optional[str], optional
        The name of the loader, by default None.
    extensions : Optional[list[str]], optional
        The extensions that the loader supports, by default None.
    wraps : Optional[Callable[..., Any]], optional
        The function to wrap, by default None.
    path_arg : Optional[str], optional
        The name of the argument that is the path, by default None.
    """
    ...


@overload
def dataloader(
    name: str,
    extensions: Optional[list[str]] = None,
    wraps: Optional[Callable[P, R]] = None,
    path_arg: Optional[str] = None,
) -> Callable[[Callable[..., Any]], DatasetLoader[P, R]]:
    """
    Decorator to register a function as a dataset loader.

    Parameters
    ----------
    name : str
        The name of the loader.
    extensions : Optional[list[str]], optional
        The extensions that the loader supports, by default None.
    wraps : Optional[Callable[..., Any]], optional
        The function to wrap, by default None.
    path_arg : Optional[str], optional
        The name of the argument that is the path, by default None.

    Returns
    -------
    Callable[[Callable[..., Any]], DatasetLoader]
        The decorator.
    """
    ...


def dataloader(
    func: Union[Callable[..., Any], str, None] = None,
    name: Optional[str] = None,
    extensions: Optional[list[str]] = None,
    wraps: Optional[Callable[..., Any]] = None,
    path_arg: Optional[str] = None,
) -> DatasetLoader:
    """
    Decorator to register a function as a dataset loader.

    Parameters
    ----------
    func : Union[Callable[..., Any], str, None], optional
        The function to decorate, by default None.
    name : Optional[str], optional
        The name of the loader, by default None.
    extensions : Optional[list[str]], optional
        The extensions that the loader supports, by default None.
    wraps : Optional[Callable[..., Any]], optional
        The function to wrap, by default None.
    path_arg : Optional[str], optional
        The name of the argument that is the path, by default None.

    Returns
    -------
    DatasetLoader
        The dataset loader.
    """
    if func is None:
        return functools.partial(
            dataloader,
            name=name,
            extensions=extensions,
            wraps=wraps,
            path_arg=path_arg,
        )
    elif isinstance(func, str):
        return functools.partial(
            dataloader,
            name=func,
            extensions=extensions,
            wraps=wraps,
            path_arg=path_arg,
        )
    loader = DatasetLoader(
        func,
        name=name,
        extensions=extensions,
        path_arg=path_arg,
        wraps=wraps,
    )
    for ext in loader.extensions:
        loaders[ext] = loader
    return func


def load_dataset(
    __path: Optional[str],
    __format: str = DEFAULT_FORMAT,
    /,
    *args,
    **kwargs,
) -> Dataset:
    """
    Load a dataset from a path.

    Parameters
    ----------
    __path : Optional[str]
        The path to the dataset.
    __format : str, optional
        The format of the dataset, by default DEFAULT_FORMAT.
    *args : tuple
        The arguments to pass to the loader.
    **kwargs : dict
        The keyword arguments to pass to the loader.

    Returns
    -------
    Dataset
        The loaded dataset.
    """
    ext = "." + __path.split(".")[-1]
    loader = loaders.get(ext)
    if loader is None:
        msg = f"unknown extension '{ext}'"
        raise ValueError(msg)
    if loader.path_arg is not None:
        kwargs[loader.path_arg] = __path
    else:
        # assume the first argument is the path
        args = (__path, *args)
    return Dataset(
        loader,
        __format,
        loader_args=args,
        loader_kwargs=kwargs,
    )
