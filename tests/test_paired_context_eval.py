# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from graph_eqa.utils.eval_context import FrozenSceneContext
from openai import OpenAI
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_paired_context_eval.py"
spec = importlib.util.spec_from_file_location("paired_eval", SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def context():
    return {
        "agent_state": "Estimated local camera pose [0, 0, 1.5]",
        "scene_graph": {
            "nodes": [
                {"id": "room_0", "name": "room"},
                {"id": "object_0", "name": "cup", "position": [1, 0, 0.7]},
            ],
            "links": [],
            "directed": True,
            "multigraph": False,
        },
        "actions": {
            "objects": [{"id": "object_0", "name": "cup"}],
            "rooms": [{"id": "room_0", "name": "room"}],
            "frontiers": [],
            "regions": [],
        },
    }


def test_context_geometry_is_bound_to_observed_node_and_snapshot_is_immutable():
    data = context()
    snapshot = FrozenSceneContext(data)
    data["scene_graph"]["nodes"][1]["position"] = [100, 100, 100]
    assert snapshot.get_position_from_id("object_0") == [1, 0, 0.7]
    with pytest.raises(ValueError, match="Not an actionable"):
        snapshot.get_position_from_id("room_0")
    bad = context()
    bad["scene_graph"]["nodes"][1]["position"] = [float("nan"), 0, 1]
    with pytest.raises(ValueError, match="finite observed"):
        FrozenSceneContext(bad)
    with pytest.raises(ValueError, match="requires only"):
        FrozenSceneContext({**context(), "answer": "B"})


def test_different_provided_models_use_their_own_endpoint_with_identical_evidence(tmp_path, monkeypatch):
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "image.png")
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "case",
                    "question": "Color?",
                    "choices": ["red", "blue"],
                    "image": "image.png",
                    "context": context(),
                }
            ]
        )
    )
    profiles = []
    for name in ("model-one", "model-two"):
        profile = tmp_path / f"{name}.yaml"
        profile.write_text(f"vlm:\n  name: {name}\n  base_url: http://{name}/v1\n  max_tokens: 128\n")
        profiles.append(f"{name}={profile}")
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body))
        reply = (
            {"answer": "B", "explanation": "blue", "confidence_level": 0.5, "is_confident": False}
            if body["response_format"]["json_schema"]["name"] == "ImageOnlyAnswer"
            else {
                "steps": [],
                "answer": {
                    "answer": "blue",
                    "explanation_ans": "blue",
                    "explanation_conf": "uncertain",
                    "confidence_level": 0.5,
                    "is_confident": False,
                },
                "image_description": "cup",
                "scene_graph_description": "cup",
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(reply)},
                    }
                ],
            },
        )

    def client(**kwargs):
        return OpenAI(
            base_url=kwargs["base_url"],
            api_key="unused",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )

    monkeypatch.setattr("graph_eqa.utils.model_api.OpenAI", client)
    monkeypatch.setattr(driver, "source_revision", lambda _: "fixture")
    assert driver.run(SimpleNamespace(cases=cases, answers=None, model_config=profiles, output=tmp_path / "run")) == 0
    assert [(host, request["model"]) for host, request in calls] == [
        (name, name) for name in ("model-one", "model-two") for _ in range(2)
    ]
    images = [
        block["image_url"]["url"]
        for _, request in calls
        for message in request["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
        if block["type"] == "image_url"
    ]
    assert len(images) == 4 and len(set(images)) == 1


@pytest.mark.parametrize("fail_first", [False, True])
def test_pair_keeps_images_identical_and_gold_out_of_model_requests(tmp_path, monkeypatch, fail_first):
    image = tmp_path / "current.png"
    Image.new("RGB", (16, 16), "blue").save(image)
    cases = tmp_path / "cases.json"
    item = {
        "id": "case",
        "question": "Which color?",
        "choices": ["red", "blue"],
        "image": "current.png",
        "context": context(),
    }
    cases.write_text(json.dumps([item]))
    answers = tmp_path / "answers.json"
    answers.write_text('{"case":"B"}')
    profile = tmp_path / "model.yaml"
    profile.write_text("vlm:\n  name: fixture\n  base_url: http://fixture/v1\n  max_tokens: 128\n")
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            reply = {"answer": "B", "explanation": "blue", "confidence_level": 0.5, "is_confident": False}
        else:
            reply = {
                "steps": [],
                "answer": {
                    "answer": "blue",
                    "explanation_ans": "blue",
                    "explanation_conf": "uncertain",
                    "confidence_level": 0.5,
                    "is_confident": False,
                },
                "image_description": "cup",
                "scene_graph_description": "cup",
            }
        text = "invalid" if fail_first and len(calls) == 1 else json.dumps(reply)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            },
        )

    client = OpenAI(
        base_url="http://fixture/v1",
        api_key="unused",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr("graph_eqa.utils.model_api.OpenAI", lambda **_: client)
    monkeypatch.setattr(driver, "source_revision", lambda _: "fixture-revision")
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == answers:
            assert len(calls) == 2, "Gold was loaded before all model calls completed"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    output = tmp_path / "run"
    rc = driver.run(SimpleNamespace(cases=cases, answers=answers, model_config=[f"fixture={profile}"], output=output))
    assert rc == int(fail_first)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["planned_rows"] == manifest["completed_rows"] == 2
    assert manifest["failed_rows"] == int(fail_first)
    images = [
        [
            block["image_url"]["url"]
            for message in request["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "image_url"
        ]
        for request in calls
    ]
    assert images[0] == images[1] and len(images[0]) == 1
    assert all(request["model"] == "fixture" and request["max_tokens"] == 128 for request in calls)
    scored = json.loads((output / "scored_results.json").read_text())
    assert scored[1]["answer"] == "B" and scored[1]["is_confident"] is False
    assert scored[1]["correct_if_stopped_here"] is True
    assert scored[0]["correct_if_stopped_here"] is (None if fail_first else True)
    assert all(row["interaction_steps"] == 0 for row in scored)
    bad_case = copy.deepcopy(item)
    bad_case["answer"] = "B"
    cases.write_text(json.dumps([bad_case]))
    with pytest.raises(ValueError, match="scoring labels are separate"):
        driver.read_cases(cases)
