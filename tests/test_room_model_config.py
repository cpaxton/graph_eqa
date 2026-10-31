# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

from pathlib import Path
from types import SimpleNamespace

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
