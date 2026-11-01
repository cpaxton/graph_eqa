# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "sensor_graph", Path(__file__).resolve().parents[1] / "scripts/serve_sensor_graph.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.mark.parametrize(
    "payload",
    [
        {"frame": "frame.npz", "frontiers": [], "answer": "B"},
        {"frame": "../frame.npz", "frontiers": []},
        {"frame": "frame.npz", "frontiers": [[float("nan"), 0, 0]]},
    ],
)
def test_worker_rejects_extra_fields_path_escape_and_invalid_frontiers_before_native_import(payload, monkeypatch):
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps(payload) + "\n"))
    with pytest.raises(ValueError):
        worker.serve(SimpleNamespace())
