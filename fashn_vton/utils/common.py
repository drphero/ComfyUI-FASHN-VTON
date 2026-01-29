"""Common Python utility functions."""

from typing import Any, Dict, Optional, Tuple


def exists(val: Any) -> bool:
    """Check if value is not None."""
    return val is not None


def default(val: Any, d: Any) -> Any:
    """Return val if not None, else default (or call it if callable)."""
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(val: Any, length: Optional[int] = None) -> Tuple:
    """Convert value to tuple with optional length validation."""
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output


def compact(input_dict: Dict) -> Dict:
    """Filter None values from dictionary."""
    return {key: value for key, value in input_dict.items() if exists(value)}
