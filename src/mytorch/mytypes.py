from pathlib import Path
from typing import TypeVar, List, Tuple, Annotated, Callable, Dict

from numpy import ndarray
from pydantic import FilePath, DirectoryPath, BeforeValidator, validate_call
from torch import Tensor

T = TypeVar("T")
type MaybeListLike[T] = List[T] | Tuple[T] | T


@validate_call
def ensure_iterable(x: MaybeListLike[T]) -> Tuple[T]:
    if not isinstance(x, (list, tuple)):
        x = (x,)
    return tuple(x)


@validate_call
def ensure_dir_exists(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


type ListLike[T] = Annotated[Tuple[T] | List[T], BeforeValidator(ensure_iterable)]
type NoInit[T] = Callable[..., T]
type CreateIfNotExistsDir = Annotated[DirectoryPath, BeforeValidator(ensure_dir_exists)]
type PathLike = FilePath | DirectoryPath | Path
type PathDict = Dict[str, PathLike | PathDict]
type Maybe[T] = T | None
type ArrayLike = List | Tensor | ndarray | Tuple
