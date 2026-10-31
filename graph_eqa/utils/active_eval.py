# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Bounded reference-controller interaction with externally supplied sensors.

This module has no simulator or scorer dependency. Observe supplies learned
context plus a fresh RGB capture; execute returns measured motion feedback.
"""

import json
from pathlib import Path

from graph_eqa.planners.vlm_planner_gpt import VLMPlannerEQAGPT
from graph_eqa.utils.eval_context import FrozenSceneContext


def run_active_episode(cfg, question, choices, observe, execute, output, *, max_decisions=8):
    if isinstance(max_decisions, bool) or not isinstance(max_decisions, int) or max_decisions <= 0:
        raise ValueError("A positive decision budget is required")
    if not isinstance(question, str) or not question.strip() or not 2 <= len(choices) <= 26:
        raise ValueError("A question and multiple-choice options are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    letters = [chr(65 + i) for i in range(len(choices))]
    prompt = question + "\n" + "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices, strict=True))
    planner, previous_stamp, rows = None, 0, []
    result = {"status": "running", "answer": None, "is_confident": False, "decisions": 0}
    try:
        for index in range(max_decisions):
            capture = observe()
            if set(capture) not in (
                {"timestamp_ns", "image", "context"},
                {"timestamp_ns", "image", "context", "images"},
            ):
                raise ValueError("Observe requires timestamp_ns, image, context and optional labeled images")
            stamp = capture["timestamp_ns"]
            if not isinstance(stamp, int) or stamp <= previous_stamp:
                raise ValueError("Assessment requires a fresh capture timestamp")
            previous_stamp = stamp
            scene = FrozenSceneContext(capture["context"])
            image = Path(capture["image"])
            if image.suffix.lower() != ".png" or not image.is_file():
                raise ValueError("Assessment requires a captured PNG")
            (output / f"current_img_{index}.png").write_bytes(image.read_bytes())
            evidence = None
            if "images" in capture:
                evidence = capture["images"]
                if not isinstance(evidence, list) or not 1 <= len(evidence) <= 16:
                    raise ValueError("Use one to sixteen captured evidence images, current first")
                saved = []
                for image_index, item in enumerate(evidence):
                    if (
                        set(item) != {"path", "label"}
                        or not isinstance(item["label"], str)
                        or not item["label"].strip()
                    ):
                        raise ValueError("Evidence requires only a captured PNG path and nonempty label")
                    source = Path(item["path"])
                    if source.suffix.lower() != ".png" or not source.is_file():
                        raise ValueError("Evidence requires a captured PNG")
                    data = source.read_bytes()
                    if image_index == 0 and data != image.read_bytes():
                        raise ValueError("The first evidence image must match the current capture")
                    destination = output / f"evidence-{index:03d}-{image_index:02d}.png"
                    destination.write_bytes(data)
                    saved.append({"path": str(destination), "label": item["label"]})
                evidence = saved
                (output / f"evidence-{index:03d}.json").write_text(json.dumps(evidence, indent=2) + "\n")
            (output / f"context-{index:03d}.json").write_text(json.dumps(capture["context"], indent=2) + "\n")
            if planner is None:
                planner = VLMPlannerEQAGPT(cfg, scene, prompt, letters, choices, output)
            else:
                planner.sg_sim = scene
            # Replace the packet on every fresh assessment, retaining the same
            # planner and its measured action history. None preserves legacy
            # current-image behavior rather than reusing an earlier packet.
            planner.image_evidence = evidence
            position, target, confident, confidence, answer = planner.get_next_action()
            row = {
                "decision": index,
                "timestamp_ns": stamp,
                "answer": answer,
                "is_confident": confident,
                "confidence_level": confidence,
                "target_id": target,
                "target_xyz": position,
                "motion": None,
            }
            # Stop and answer exactly as the reference model decided. A prior
            # answer, retrieval score or admission rule cannot replace it.
            result.update(answer=answer, is_confident=bool(confident), decisions=index + 1)
            if confident:
                result["status"] = "model_stopped"
            elif target is None:
                result["status"] = "no_action_not_confident"
            elif index + 1 == max_decisions:
                result["status"] = "decision_budget_exhausted"
            else:
                receipt = execute(target, position)
                if not isinstance(receipt, dict) or not isinstance(receipt.get("success"), bool):
                    raise ValueError("Execution must return an explicit measured success/failure receipt")
                row["motion"] = receipt
                planner.record_motion_result(receipt)
            rows.append(row)
            with (output / "decisions.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if result["status"] != "running":
                break
    except Exception as exc:
        result.update(status="infrastructure_or_contract_failure", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["completed_decisions"] = len(rows)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
