# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Read-only reference-controller context for paired model evaluation.

This adapter makes no model, simulator or navigation calls. Geometry stays tied
to the observed graph node selected by its instance ID.
"""

import copy
import json
import math


class FrozenSceneContext:
    def __init__(self, context):
        if set(context) != {"agent_state", "scene_graph", "actions"}:
            raise ValueError("Context requires only agent_state, scene_graph and actions")
        self.context = copy.deepcopy(context)
        context = self.context
        if not isinstance(context["agent_state"], str) or not context["agent_state"].strip():
            raise ValueError("Missing estimated agent state")
        graph = context["scene_graph"]
        self.nodes = {node["id"]: node for node in graph["nodes"]}
        if len(self.nodes) != len(graph["nodes"]):
            raise ValueError("Duplicate graph node IDs")
        actions = context["actions"]
        if set(actions) != {"objects", "rooms", "frontiers", "regions"}:
            raise ValueError("Invalid controller action registry")
        for kind in ("objects", "rooms"):
            if any(set(item) != {"id", "name"} for item in actions[kind]):
                raise ValueError("Action records require only id and name")
        self.object_node_ids = [item["id"] for item in actions["objects"]]
        self.object_node_names = [item["name"] for item in actions["objects"]]
        self.room_node_ids = [item["id"] for item in actions["rooms"]]
        self.room_node_names = [item["name"] for item in actions["rooms"]]
        self.frontier_node_ids = list(actions["frontiers"])
        self.region_node_ids = list(actions["regions"])
        ids = self.object_node_ids + self.room_node_ids + self.frontier_node_ids
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or i not in self.nodes for i in ids):
            raise ValueError("Action IDs must be unique observed graph nodes")
        for item in actions["objects"] + actions["rooms"]:
            if not isinstance(item["name"], str) or item["name"] != self.nodes[item["id"]].get("name"):
                raise ValueError("Action labels must match their observed graph node")
        for target in self.object_node_ids + self.frontier_node_ids:
            position = self.nodes[target].get("position", [])
            if len(position) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position):
                raise ValueError(f"Missing finite observed target geometry: {target}")

    @property
    def scene_graph_str(self):
        return json.dumps(self.context["scene_graph"])

    def get_current_semantic_state_str(self):
        return self.context["agent_state"]

    def get_position_from_id(self, target):
        if target not in self.object_node_ids + self.frontier_node_ids:
            raise ValueError(f"Not an actionable observed target: {target}")
        return list(self.nodes[target]["position"])
