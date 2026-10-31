# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

import json
from enum import Enum
from types import SimpleNamespace

import httpx
import pytest
from graph_eqa.planners.vlm_planner_gpt import VLMPlannerEQAGPT, create_planner_response
from graph_eqa.utils.model_api import ModelResponseError, StructuredModelAPI, with_model_config
from omegaconf import OmegaConf
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel, ValidationError


class Choice(BaseModel):
    answer: str


def provider(tmp_path, *, mode="json_schema", content='{"answer":"B"}', finish="stop", refusal=None):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "served-qwen",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "refusal": refusal,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )

    client = OpenAI(
        base_url="http://fixture/v1",
        api_key="secret-test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    api = StructuredModelAPI(
        {"name": "requested-qwen", "response_mode": mode, "max_tokens": 256}, tmp_path, client=client
    )
    return api, calls


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "text"])
def test_modes_preserve_history_images_and_record_actual_request_response(tmp_path, mode):
    api, calls = provider(tmp_path, mode=mode)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID"}}
    messages = [
        {"role": "system", "content": "planner"},
        {"role": "assistant", "content": "previous step"},
        {"role": "user", "content": [image]},
    ]
    assert api.complete(messages, Choice).answer == "B"
    assert len(messages) == 3
    assert calls[0]["messages"][-3:] == messages
    assert calls[0]["model"] == "requested-qwen"
    assert calls[0]["max_tokens"] == 256
    if mode == "text":
        assert "response_format" not in calls[0]
    else:
        assert calls[0]["response_format"]["type"] == mode
    trace_text = (tmp_path / "model_calls.jsonl").read_text()
    trace = json.loads(trace_text)
    assert trace["validated"] is True
    assert trace["response"]["model"] == "served-qwen"
    assert trace["response"]["usage"]["total_tokens"] == 13
    assert "secret-test-key" not in trace_text


def test_text_schema_retains_the_structured_contract_but_shows_root_before_definitions(tmp_path):
    class Nested(BaseModel):
        choice: Choice

    schemas = {}
    for mode in ("text", "json_schema"):
        api, calls = provider(tmp_path / mode, mode=mode, content='{"choice":{"answer":"B"}}')
        api.complete([{"role": "user", "content": "choose"}], Nested)
        request = calls[0]
        schemas[mode] = (
            json.loads(request["messages"][0]["content"].split("\n", 1)[1])
            if mode == "text"
            else request["response_format"]["json_schema"]["schema"]
        )
    assert schemas["text"] == schemas["json_schema"]
    assert list(schemas["text"])[0] == "type" and list(schemas["text"])[-1] == "$defs"


@pytest.mark.parametrize(
    "content,finish,refusal",
    [
        ('{"answer":"B"}', "length", None),
        (None, "stop", "refused"),
        ("", "stop", None),
        ('{"wrong":"field"}', "stop", None),
        ("not JSON", "stop", None),
    ],
)
def test_invalid_responses_are_saved_and_fail_without_changing_model_or_mode(tmp_path, content, finish, refusal):
    api, calls = provider(tmp_path, content=content, finish=finish, refusal=refusal)
    with pytest.raises((ModelResponseError, ValidationError)):
        api.complete([{"role": "user", "content": "question"}], Choice)
    assert len(calls) == 1
    trace = json.loads((tmp_path / "model_calls.jsonl").read_text())
    assert trace["validated"] is False
    assert trace["response"]["choices"][0]["message"]["content"] == content


def test_model_profile_cannot_change_dataset_or_perception(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("vlm:\n  name: qwen\ndata:\n  use_semantic_data: true\n")
    with pytest.raises(ValueError, match="only a vlm mapping"):
        with_model_config(OmegaConf.create({"data": {"use_semantic_data": False}}), path)
    path.write_text("vlm:\n  name: qwen\n  provider: openai_compatible\n")
    config = with_model_config(
        OmegaConf.create({"data": {"use_semantic_data": False}, "vlm": {"room_model": "gpt-4o-mini"}}), path
    )
    assert not config.data.use_semantic_data
    assert config.vlm.name == "qwen"
    assert config.vlm.room_model is None


@pytest.mark.parametrize("target_id", ["object_2", "object_99"])
@pytest.mark.parametrize("external_ids", [False, True])
def test_native_planner_preserves_duplicate_object_identity_and_model_answer(tmp_path, target_id, external_ids):
    object_ids = ["object-1", "2-object"] if external_ids else ["object_1", "object_2"]
    room_ids = ["room-1", "2-room"] if external_ids else ["room_1", "room_2"]
    valid = target_id == "object_2"
    target_id = object_ids[1] if valid else "object_99"
    response = {
        "steps": [
            {
                "explanation_room": "inspect",
                "explanation_obj": "second cup",
                "room_id": room_ids[1],
                "object_id": target_id,
            }
        ],
        "answer": {
            "explanation_ans": "blue",
            "answer": "blue",
            "explanation_conf": "uncertain",
            "confidence_level": 0.5,
            "is_confident": False,
        },
        "scene_graph_description": "two cups",
    }
    api, calls = provider(tmp_path, content=json.dumps(response))
    scene = SimpleNamespace(
        object_node_ids=object_ids,
        object_node_names=["cup", "cup"],
        room_node_ids=room_ids,
        room_node_names=["room", "room"],
        frontier_node_ids=["3-frontier"] if external_ids else [],
        region_node_ids=["4-region"] if external_ids else [],
        scene_graph_str="two different cup instances",
        get_current_semantic_state_str=lambda: "estimated local pose",
        get_position_from_id=lambda target: {object_ids[0]: [1, 0, 0], object_ids[1]: [2, 0, 0]}[target],
    )
    cfg = OmegaConf.create({"name": "qwen", "base_url": "http://unused/v1", "use_image": False, "add_history": True})
    planner = VLMPlannerEQAGPT(cfg, scene, "Which color?", ["A", "B"], ["red", "blue"], tmp_path)
    planner.model_api = api
    if not valid:
        with pytest.raises(ValidationError):
            planner.get_next_action()
        assert planner.t == 0 and planner._history == ""
        return
    pose, target, confident, _, answer = planner.get_next_action()
    assert pose == [2, 0, 0] and target == object_ids[1]
    assert answer == "B" and confident is False
    assert "object name: cup" in planner._history
    assert planner.t == 1
    assert calls[0]["response_format"]["json_schema"]["schema"]["$defs"]["object_node_list"]["enum"] == object_ids


@pytest.mark.parametrize("use_image", [False, True])
@pytest.mark.parametrize("has_frontiers", [False, True])
def test_planner_strict_schema_closes_all_objects_and_keeps_empty_plans_legal(use_image, has_frontiers):
    def enum(name, values):
        return Enum(name, {value: value for value in values}, type=str)
    response = create_planner_response(
        enum("Frontiers", ["frontier_1"]) if has_frontiers else None,
        enum("Rooms", ["room_1"]),
        enum("Regions", ["region_1"]),
        enum("Objects", ["object_1"]),
        enum("Answers", ["red", "blue"]),
        use_image=use_image,
    )
    schema = pydantic_function_tool(response)["function"]["parameters"]
    assert list(schema["properties"])[:2] == ["answer", "steps"]
    assert schema["required"][:2] == ["answer", "steps"]

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)
    data = {
        "steps": [],
        "answer": {
            "answer": "blue",
            "explanation_ans": "unknown",
            "explanation_conf": "uncertain",
            "confidence_level": 0.0,
            "is_confident": False,
        },
        "scene_graph_description": "observed objects",
    }
    if use_image:
        data["image_description"] = "current view"
    assert response.model_validate(data).steps == []
