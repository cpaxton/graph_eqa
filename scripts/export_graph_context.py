#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Export controller context from a saved learned Hydra graph, without inference.

Requires the native Spark DSG runtime. No simulator, labels, enrichment file or
question is read. Generic observed room names are retained; room inference and
image ranking are outside this frozen-context export.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import spark_dsg
from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    graph = spark_dsg.DynamicSceneGraph.load(str(args.graph))
    scene = SceneGraphSim(cfg, args.output.parent, SimpleNamespace(graph=graph), device="cpu", enrich_object_labels="")
    scene.update()
    args.output.write_text(json.dumps(scene.planner_context(), indent=2) + "\n")


if __name__ == "__main__":
    main()
