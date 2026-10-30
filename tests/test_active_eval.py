# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

import json

import pytest
from graph_eqa.utils.active_eval import run_active_episode
from PIL import Image
from test_paired_context_eval import context


def test_measured_failure_and_new_context_reach_same_controller_without_answer_override(tmp_path, monkeypatch):
    image = tmp_path / "capture.png"
    Image.new("RGB", (4, 4), "red").save(image)
    planners, seen, receipts = [], [], []

    class Planner:
        def __init__(self, cfg, scene, *args):
            self.sg_sim = scene
            planners.append(self)

        def get_next_action(self):
            seen.append(self.sg_sim.get_current_semantic_state_str())
            return ([1, 0, 0.7], "object_0", False, 0.2, "A") if len(seen) == 1 else (None, None, True, 0.9, "B")

        def record_motion_result(self, receipt):
            receipts.append(receipt)

    monkeypatch.setattr("graph_eqa.utils.active_eval.VLMPlannerEQAGPT", Planner)
    observations = iter([1, 2])

    def observe():
        stamp = next(observations)
        ctx = context()
        ctx["agent_state"] = f"capture {stamp}"
        return {"timestamp_ns": stamp, "image": image, "context": ctx}

    motion = {"success": False, "reason": "occupied_footprint", "current_pose": [0, 0, 0]}
    result = run_active_episode({}, "Color?", ["red", "blue"], observe, lambda *args: motion, tmp_path / "run")
    assert result["answer"] == "B" and result["status"] == "model_stopped"
    assert len(planners) == 1 and seen == ["capture 1", "capture 2"] and receipts == [motion]
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    assert rows[0]["motion"] == motion and rows[1]["timestamp_ns"] > rows[0]["timestamp_ns"]


def test_stale_capture_stops_before_second_model_call_and_retains_failure(tmp_path, monkeypatch):
    image = tmp_path / "capture.png"
    Image.new("RGB", (4, 4)).save(image)
    calls = []

    class Planner:
        def __init__(self, *args):
            pass

        def get_next_action(self):
            calls.append(1)
            return [1, 0, 0.7], "object_0", False, 0.2, "A"

        def record_motion_result(self, receipt):
            pass

    monkeypatch.setattr("graph_eqa.utils.active_eval.VLMPlannerEQAGPT", Planner)

    def observe():
        return {"timestamp_ns": 1, "image": image, "context": context()}

    with pytest.raises(ValueError, match="fresh capture"):
        run_active_episode({}, "Color?", ["red", "blue"], observe, lambda *args: {"success": True}, tmp_path / "run")
    assert len(calls) == 1
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["status"] == "infrastructure_or_contract_failure" and result["completed_decisions"] == 1
