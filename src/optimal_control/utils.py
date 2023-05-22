from typing import Any


def exists(x: Any) -> bool:
    return x is not None


def default(x: Any, y: Any) -> Any:
    return x if x is not None else y
