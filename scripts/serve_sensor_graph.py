#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Persistent native Hydra frontend for externally captured, learned RGB-D.

Run in the isolated native dependency container. JSONL stdin requests reference
new packets under --inputs; replies are atomically written under --output so
native stdout logging cannot corrupt the protocol. No simulator/model/dataset
is imported. The caller must establish online segmentation provenance.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def create_pipeline(labels_path, packet, output):
    import hydra_python as hydra
    from hydra_python._hydra_bindings import PythonConfig

    labels = yaml.safe_load(labels_path.read_text())
    configs = PythonConfig()
    for namespace in ("frontend", "backend", "reconstruction"):
        configs.add_file(hydra.get_config_path() / "habitat" / f"{namespace}_config.yaml", config_ns=namespace)
    configs.add_file(labels_path)
    configs.add_yaml(
        yaml.safe_dump(
            {
                "frontend": {"type": "FrontendModule", "objects": {"bounding_box_type": "AABB"}},
                "backend": {"type": "BackendModule"},
                "lcd": {"lcd_use_bow_vectors": False},
                "reconstruction": {
                    "type": "ReconstructionModule",
                    "show_stats": False,
                    "pose_graphs": {"make_pose_graph": True},
                },
            }
        )
    )
    hydra.set_glog_level(0, 0)
    config = hydra.PipelineConfig(configs)
    config.enable_reconstruction = True
    config.label_names = {entry["label"]: entry["name"] for entry in labels["label_names"]}
    colormap = hydra.SegmentationColormap.from_names(names=list(config.label_names.values()))
    colormap.fill_label_space(config.label_space)
    config.logs.log_dir = str(output)
    pipeline = hydra.HydraPipeline(config, robot_id=0, config_verbosity=0, freeze_global_info=False)
    k, (height, width) = packet["camera_K"], packet["depth"].shape
    pipeline.init(
        configs,
        hydra.create_camera(
            {
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "width": width,
                "height": height,
            }
        ),
    )
    cadence = yaml.safe_load((hydra.get_config_path() / "habitat/reconstruction_config.yaml").read_text())[
        "num_poses_per_update"
    ]
    return pipeline, int(cadence), labels["total_semantic_labels"]


def scene_context(pipeline, output, frontiers, packet):
    from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "scene_graph_sim": {
                "enrich_rooms": False,
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
    scene = SceneGraphSim(cfg, output, pipeline, device="cpu", enrich_object_labels="")
    scene.update(frontier_nodes=np.asarray(frontiers, dtype=float).reshape(-1, 3))
    context = scene.planner_context()
    # Native agent nodes update at the reconstruction cadence. Expose the
    # actual capture pose separately, rather than claiming that node is fresh.
    context["agent_state"] += (
        f"\nLatest RGB-D camera-to-local-world pose at timestamp_ns={int(packet['timestamp_ns'])}: "
        + json.dumps(packet["camera_pose"].tolist())
        + ". The native agent node may precede this capture. Learned object labels are unverified predictions."
        + " Frontier IDs refer to this capture's registry; their positions can change on subsequent captures."
    )
    return context


def serve(args):
    from scipy.spatial.transform import Rotation

    pipeline, cadence, labels = None, None, None
    count, previous_stamp = 0, 0
    calibration, image_shape = None, None
    for line in sys.stdin:
        request = json.loads(line)
        if set(request) != {"frame", "frontiers"}:
            raise ValueError("Only a fresh sensor packet and observed frontier coordinates are accepted")
        frontiers = np.asarray(request["frontiers"], dtype=float)
        if frontiers.size and (frontiers.ndim != 2 or frontiers.shape[1] != 3 or not np.isfinite(frontiers).all()):
            raise ValueError("Frontiers must be finite local XYZ positions")
        name = request["frame"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("Packet must be a local .npz basename")
        with np.load(args.inputs / name, allow_pickle=False) as data:
            packet = dict(data)
        # Contract is mounted independently from the host's agent dependencies.
        from grapheqa_sensor import validate_packet

        if pipeline is None:
            declared_labels = yaml.safe_load(args.labels.read_text())["total_semantic_labels"]
            validate_packet(packet, num_labels=declared_labels)
            pipeline, cadence, labels = create_pipeline(args.labels, packet, args.output)
            calibration, image_shape = packet["camera_K"].copy(), packet["depth"].shape
        validate_packet(packet, num_labels=labels)
        if not np.array_equal(calibration, packet["camera_K"]) or packet["depth"].shape != image_shape:
            raise ValueError("Camera calibration changed within the stream")
        stamp = int(packet["timestamp_ns"])
        if stamp <= previous_stamp:
            raise ValueError("Sensor timestamps must increase strictly")
        previous_stamp = stamp
        pose = packet["camera_pose"]
        quaternion = np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1)
        updated = pipeline.step(
            stamp, pose[:3, 3].copy(), quaternion, packet["depth"], packet["learned_labels"], packet["rgb"]
        )
        count += 1
        if bool(updated) != (count % cadence == 0):
            raise RuntimeError("Native graph update diverged from its declared cadence")
        reply = {
            "frames": count,
            "timestamp_ns": stamp,
            "graph_updated": bool(updated),
            "context": scene_context(pipeline, args.output, request["frontiers"], packet)
            if count >= 2 * cadence
            else None,
        }
        if updated:
            pipeline.graph.save(str(args.output / "graph.json"), False)
        target = args.output / (Path(name).stem + ".json")
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(reply) + "\n")
        temp.replace(target)
    if pipeline is not None:
        pipeline.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    serve(parser.parse_args())
