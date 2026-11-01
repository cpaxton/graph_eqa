# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from graph_eqa.scene_graph import scene_graph_sim as scene_module
from graph_eqa.utils.model_api import with_model_config
from omegaconf import OmegaConf


def mapping_config(enrich_rooms):
    return OmegaConf.create(
        {
            "scene_graph_sim": {
                "enrich_rooms": enrich_rooms,
                "save_image": False,
                "include_regions": False,
                "no_scene_graph": False,
                "enrich_frontiers": True,
                "key_frame_selection": {
                    "choose_final_image": True,
                    "use_clip_for_images": False,
                    "use_siglip_for_images": False,
                },
            }
        }
    )


def test_mapping_without_room_inference_requires_no_provider(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Mapping-only construction tried to create a provider")

    monkeypatch.setattr(scene_module, "StructuredModelAPI", forbidden)
    scene = scene_module.SceneGraphSim(mapping_config(False), tmp_path, SimpleNamespace(graph=None))
    assert scene.room_model_api is None


@pytest.mark.parametrize("vlm", [None, {}, "not-a-model-config", {"name": ""}])
def test_room_inference_without_model_fails_clearly_before_creating_output(tmp_path, vlm):
    cfg = mapping_config(True)
    if vlm is not None:
        cfg.vlm = vlm
    output = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="Room enrichment requires cfg.vlm.name"):
        scene_module.SceneGraphSim(cfg, output, SimpleNamespace(graph=None))
    assert not output.exists()


def test_room_client_is_lazy_and_reused_with_explicit_primary_inheritance(tmp_path, monkeypatch):
    cfg = mapping_config(True)
    cfg.vlm = {"name": "provided-model", "room_model": None, "base_url": "http://provided/v1"}
    calls = []

    def build(config, output, *, purpose, model):
        calls.append((config.name, purpose, model))
        return object()

    monkeypatch.setattr(scene_module, "StructuredModelAPI", build)
    scene = scene_module.SceneGraphSim(cfg, tmp_path, SimpleNamespace(graph=None))
    assert calls == []
    assert scene._get_room_model_api() is scene._get_room_model_api()
    assert calls == [("provided-model", "room", None)]


@pytest.mark.parametrize("room", ["omitted", None, "separate-room-model"])
def test_model_overlay_preserves_inputs_and_documents_room_override_semantics(tmp_path, room):
    data = {"vlm": {"name": "primary"}}
    if room != "omitted":
        data["vlm"]["room_model"] = room
    path = tmp_path / "profile.yaml"
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(data)))
    original_file = path.read_bytes()
    base = OmegaConf.create({"vlm": {"name": "old", "room_model": "legacy"}})
    before = OmegaConf.to_container(base)
    merged = with_model_config(base, path)
    assert merged.vlm.room_model == ("separate-room-model" if room == "separate-room-model" else None)
    assert path.read_bytes() == original_file and OmegaConf.to_container(base) == before


def test_all_shipped_room_enrichment_configs_have_explicit_model_settings():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "cfg").glob("grapheqa*.yaml"):
        cfg = OmegaConf.load(path)
        if cfg.scene_graph_sim.enrich_rooms:
            assert cfg.vlm.name and cfg.vlm.room_model


def test_planning_frontiers_match_current_action_registry_without_changing_native_map(tmp_path):
    def node(category, index, position, layer):
        return SimpleNamespace(
            id=SimpleNamespace(category=category, category_id=index),
            layer=layer,
            attributes=SimpleNamespace(position=np.array(position, dtype=float), semantic_label=0),
            timestamp=100,
        )

    native_frontier = node("f", 1800, [4, 5, 2], 3)
    graph = SimpleNamespace(
        dynamic_layers=[SimpleNamespace(nodes=[node("a", 0, [0, 0, 1.5], 2)])],
        nodes=[native_frontier, node("r", 0, [0, 0, 0], 4)],
        edges=[],
        dynamic_interlayer_edges=[],
    )
    scene = scene_module.SceneGraphSim(mapping_config(False), tmp_path, SimpleNamespace(graph=graph))
    scene.update(frontier_nodes=np.array([[0.3, 1.3, 0], [0.4, -1, 0]]))
    context = scene.planner_context()
    frontiers = {item["id"] for item in context["scene_graph"]["nodes"] if item["name"] == "frontier"}
    assert frontiers == set(context["actions"]["frontiers"]) == {"frontier_0", "frontier_1"}
    assert graph.nodes[0] is native_frontier  # The native reconstruction is untouched.

    scene.update_frontier_nodes(np.array([[0.7, 1.5, 0]]))
    assert scene.frontier_node_ids == ["frontier_0"] and "frontier_1" not in scene.filtered_netx_graph
    assert scene.get_position_from_id("frontier_0").tolist() == [0.7, 1.5, 0]
    scene.update_frontier_nodes(np.empty((0, 3)))
    assert scene.frontier_node_ids == []
    assert not any(attr["name"] == "frontier" for _, attr in scene.filtered_netx_graph.nodes(data=True))
