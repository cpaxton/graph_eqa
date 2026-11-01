# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

import base64
import json
from types import SimpleNamespace

import pytest
from graph_eqa.utils.active_eval import run_active_episode
from PIL import Image
from test_model_api import provider
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


def test_native_planner_refreshes_independent_images_and_retains_motion_history(tmp_path, monkeypatch):
    images = [tmp_path / f"capture-{index}.png" for index in range(3)]
    for path, color in zip(images, ("red", "blue", "green"), strict=True):
        Image.new("RGB", (4, 4), color).save(path)
    reply = {
        "answer": {
            "answer": "red",
            "explanation_ans": "Need another view",
            "explanation_conf": "Incomplete",
            "confidence_level": 0.2,
            "is_confident": False,
        },
        "steps": [
            {
                "object_id": "object_0",
                "room_id": "room_0",
                "explanation_room": "Same room",
                "explanation_obj": "Inspect cup",
            }
        ],
        "image_description": "Captured views",
        "scene_graph_description": "Learned cup",
    }
    api, calls = provider(tmp_path / "traces", mode="text", content=json.dumps(reply))
    monkeypatch.setattr("graph_eqa.planners.vlm_planner_gpt.StructuredModelAPI", lambda *args: api)
    observations = iter((1, 2))

    def observe():
        stamp = next(observations)
        return {
            "timestamp_ns": stamp,
            "image": images[stamp],
            "context": context(),
            "images": [
                {"path": images[stamp], "label": f"CURRENT sequence={stamp}"},
                {"path": images[0], "label": "HISTORY sequence=0"},
            ],
        }

    receipt = {"success": True, "measured_pose": [0.2, 0, 0]}
    result = run_active_episode(
        SimpleNamespace(name="qwen", use_image=True, add_history=True),
        "Color?",
        ["red", "blue"],
        observe,
        lambda *args: receipt,
        tmp_path / "run",
        max_decisions=2,
    )
    assert result["status"] == "decision_budget_exhausted" and len(calls) == 2
    for index, call in enumerate(calls, 1):
        content = call["messages"][-1]["content"]
        assert [part["text"] for part in content if part["type"] == "text"] == [
            f"CURRENT sequence={index}",
            "HISTORY sequence=0",
        ]
        data = [
            base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            for part in content
            if part["type"] == "image_url"
        ]
        assert data == [images[index].read_bytes(), images[0].read_bytes()]
    assert "MOTION RESULT" not in str(calls[0]["messages"])
    assert json.dumps(receipt) in str(calls[1]["messages"])
    assert (tmp_path / "run/evidence-001.json").is_file()


def test_history_packet_cannot_replace_current_capture(tmp_path):
    current, stale = tmp_path / "current.png", tmp_path / "stale.png"
    Image.new("RGB", (4, 4), "red").save(current)
    Image.new("RGB", (4, 4), "blue").save(stale)
    with pytest.raises(ValueError, match="match the current"):
        run_active_episode(
            {},
            "Color?",
            ["red", "blue"],
            lambda: {
                "timestamp_ns": 1,
                "image": current,
                "context": context(),
                "images": [{"path": stale, "label": "CURRENT"}],
            },
            lambda *args: {"success": True},
            tmp_path / "run",
        )


def test_uncertain_empty_plan_can_be_followed_by_model_selected_motion(tmp_path, monkeypatch):
    image = tmp_path / "capture.png"
    Image.new("RGB", (4, 4)).save(image)
    decisions = iter(
        [(None, None, False, 0.1, "A"), ([1, 0, 0.7], "object_0", False, 0.2, "A"), (None, None, True, 0.9, "B")]
    )
    planners, motions = [], []

    class Planner:
        def __init__(self, *args):
            planners.append(self)

        def get_next_action(self):
            return next(decisions)

        def record_motion_result(self, receipt):
            assert receipt["success"]

    monkeypatch.setattr("graph_eqa.utils.active_eval.VLMPlannerEQAGPT", Planner)
    stamps = iter([1, 2, 3])

    def execute(target, position):
        motions.append((target, position))
        return {"success": True, "current_pose": [1, 0, 0]}

    result = run_active_episode(
        {},
        "Color?",
        ["red", "blue"],
        lambda: {"timestamp_ns": next(stamps), "image": image, "context": context()},
        execute,
        tmp_path / "run",
        max_decisions=3,
    )
    assert result["answer"] == "B" and result["status"] == "model_stopped"
    assert result["completed_decisions"] == 3 and len(planners) == 1
    assert motions == [("object_0", [1, 0, 0.7])]
    rows = [json.loads(s) for s in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    assert rows[0]["motion"] is None and rows[1]["motion"]["success"] and rows[2]["motion"] is None


def test_native_empty_plans_keep_history_and_stop_at_declared_budget(tmp_path, monkeypatch):
    image = tmp_path / "capture.png"
    Image.new("RGB", (4, 4)).save(image)
    reply = {
        "steps": [],
        "answer": {
            "answer": "red",
            "explanation_ans": "Unknown",
            "explanation_conf": "Need observations",
            "confidence_level": 0.1,
            "is_confident": False,
        },
        "image_description": "A room",
        "scene_graph_description": "Learned objects",
    }
    api, calls = provider(tmp_path / "traces", mode="text", content=json.dumps(reply))
    monkeypatch.setattr("graph_eqa.planners.vlm_planner_gpt.StructuredModelAPI", lambda *args: api)
    stamps = iter([1, 2])

    def unexpected_motion(*args):
        pytest.fail("An empty plan must never force motion")

    result = run_active_episode(
        SimpleNamespace(name="qwen", use_image=True, add_history=True),
        "Color?",
        ["red", "blue"],
        lambda: {"timestamp_ns": next(stamps), "image": image, "context": context()},
        unexpected_motion,
        tmp_path / "run",
        max_decisions=2,
    )
    assert result["status"] == "no_action_not_confident" and result["completed_decisions"] == 2
    assert len(calls) == 2
    second = str(calls[1]["messages"])
    assert "No observation-gathering action selected; no motion executed" in second
    assert "Answer(t=0): A red" in second and "Confident: False" in second and "At t = 1" in second


def test_interrupted_execution_keeps_decision_without_fabricating_motion_receipt(tmp_path, monkeypatch):
    image = tmp_path / "capture.png"
    Image.new("RGB", (4, 4)).save(image)

    class Planner:
        def __init__(self, *args):
            pass

        def get_next_action(self):
            return [1, 0, 0.7], "object_0", False, 0.2, "A"

    monkeypatch.setattr("graph_eqa.utils.active_eval.VLMPlannerEQAGPT", Planner)

    def execute(*args):
        raise OSError("recording reserve reached after motion began")

    with pytest.raises(OSError, match="recording reserve"):
        run_active_episode(
            {},
            "Color?",
            ["red", "blue"],
            lambda: {"timestamp_ns": 1, "image": image, "context": context()},
            execute,
            tmp_path / "run",
        )
    row = json.loads((tmp_path / "run/decisions.jsonl").read_text())
    assert row["target_id"] == "object_0" and row["execution_attempted"]
    assert row["motion"] is None and "recording reserve" in row["execution_error"]
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["status"] == "infrastructure_or_contract_failure" and result["completed_decisions"] == 1
