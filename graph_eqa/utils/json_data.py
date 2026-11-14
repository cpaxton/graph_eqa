# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Plain finite JSON numbers at external execution and prompt boundaries."""

import json

import numpy as np


def _numpy_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def json_numeric_copy(value):
    """Own numeric arrays/scalars; reject unknown objects and nonfinite values."""
    return json.loads(json.dumps(value, default=_numpy_default, allow_nan=False))
