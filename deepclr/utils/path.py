import os.path as osp
from typing import Any, List, Optional, TypeVar


_T = TypeVar('_T', Optional[str], Optional[List[Any]])


def expand_path(path: _T) -> _T:
    """Expand single path or list of paths to absolute paths."""
    if path is not None:
        if isinstance(path, list):
            path = [expand_path(p) for p in path]
        else:
            path = osp.realpath(osp.expandvars(osp.expanduser(path)))
            if '%' in path or '$' in path:
                raise RuntimeError(f"Could not replace a variable in path '{path}'")
    return path
