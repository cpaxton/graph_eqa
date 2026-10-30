#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Pair model-only and reference-graph requests on identical captured RGB evidence.

This is frozen-context inference, not active navigation or episode success.
Models run serially. Each case has one image file, which may itself be a saved
reference image packet. Scoring labels are loaded only after all model calls.
"""

import argparse
import base64
import hashlib
import json
import subprocess
import time
from enum import Enum
from pathlib import Path

from graph_eqa.planners.vlm_planner_gpt import VLMPlannerEQAGPT
from graph_eqa.utils.eval_context import FrozenSceneContext
from graph_eqa.utils.model_api import StructuredModelAPI, with_model_config
from omegaconf import OmegaConf
from pydantic import create_model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_revision(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("diff", "HEAD", "--stat") or git("ls-files", "--others", "--exclude-standard"):
        raise RuntimeError("Paired evaluation requires a clean committed checkout, including untracked source")
    return git("rev-parse", "HEAD")


def read_cases(path):
    cases = json.loads(path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty list")
    seen = set()
    for case in cases:
        if set(case) != {"id", "question", "choices", "image", "context"}:
            raise ValueError("Cases allow only id, question, choices, image and context; scoring labels are separate")
        if not isinstance(case["id"], str) or not case["id"].strip():
            raise ValueError("Case IDs must be nonempty strings")
        if case["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["id"])
        if (
            not isinstance(case["question"], str)
            or not case["question"].strip()
            or not isinstance(case["choices"], list)
            or not 2 <= len(case["choices"]) <= 26
            or not all(isinstance(choice, str) and choice.strip() for choice in case["choices"])
        ):
            raise ValueError("Missing question or invalid choice count")
        FrozenSceneContext(case["context"])
        image = (path.parent / case["image"]).resolve()
        if not image.is_file() or image.suffix.lower() != ".png":
            raise ValueError("Each case needs one captured PNG image or reference image packet")
        case["image"] = image
    return cases


def question_text(case):
    return case["question"] + "\n" + "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(case["choices"]))


def run_one(case, cfg, condition, output):
    letters = [chr(65 + i) for i in range(len(case["choices"]))]
    if condition == "reference_graph":
        (output / "current_img_0.png").write_bytes(case["image"].read_bytes())
        planner = VLMPlannerEQAGPT(
            cfg.vlm, FrozenSceneContext(case["context"]), question_text(case), letters, case["choices"], output
        )
        position, target_id, confident, confidence, answer = planner.get_next_action()
        return {
            "answer": answer,
            "is_confident": confident,
            "confidence_level": confidence,
            "proposed_target_id": target_id,
            "proposed_target_xyz": position,
        }
    choices = Enum("AnswerLetter", {letter: letter for letter in letters}, type=str)
    response_type = create_model(
        "ImageOnlyAnswer",
        answer=(choices, ...),
        explanation=(str, ...),
        is_confident=(bool, ...),
        confidence_level=(float, ...),
    )
    data = base64.b64encode(case["image"].read_bytes()).decode("ascii")
    messages = [
        {
            "role": "system",
            "content": "Answer the multiple-choice question using the attached visual evidence. "
            "Report whether the image provides enough evidence to be confident. Choose one listed answer letter.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_text(case)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
            ],
        },
    ]
    answer = StructuredModelAPI(cfg.vlm, output, purpose="model_only").complete(messages, response_type)
    return {
        "answer": answer.answer.value,
        "is_confident": answer.is_confident,
        "confidence_level": answer.confidence_level,
        "proposed_target_id": None,
        "proposed_target_xyz": None,
    }


def run(args):
    root = Path(__file__).resolve().parents[1]
    revision = source_revision(root)
    cases = read_cases(args.cases)
    profiles = []
    for value in args.model_config:
        name, path = value.split("=", 1)
        cfg = with_model_config(OmegaConf.create({"vlm": {"use_image": True, "add_history": True}}), Path(path))
        OmegaConf.resolve(cfg)
        if not cfg.vlm.use_image:
            raise ValueError("Paired context evaluation requires the same image in both conditions")
        profiles.append((name, cfg, Path(path)))
    if len({name for name, _, _ in profiles}) != len(profiles):
        raise ValueError("Model profile names must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": "frozen_context_not_active_eqa",
        "status": "running",
        "source_commit": revision,
        "driver_sha256": sha256(Path(__file__).resolve()),
        "cases_sha256": sha256(args.cases),
        "planned_rows": len(cases) * len(profiles) * 2,
        "profiles": [
            {
                "label": name,
                "sha256": sha256(path),
                "model": cfg.vlm.name,
                "base_url": cfg.vlm.get("base_url"),
                "response_mode": cfg.vlm.get("response_mode", "json_schema"),
            }
            for name, cfg, path in profiles
        ],
        "image_sha256": {case["id"]: sha256(case["image"]) for case in cases},
        "scoring_complete": args.answers is None,
    }
    rows = []
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for case in cases:
            for name, cfg, _ in profiles:
                for condition in ("model_only", "reference_graph"):
                    output = args.output / f"request-{len(rows):04d}"
                    output.mkdir()
                    row = {
                        "case_id": case["id"],
                        "profile": name,
                        "model": cfg.vlm.name,
                        "condition": condition,
                        "interaction_steps": 0,
                    }
                    started = time.monotonic()
                    try:
                        row.update(run_one(case, cfg, condition, output), complete=True)
                    except Exception as exc:
                        row.update(complete=False, error=f"{type(exc).__name__}: {exc}")
                    row["wall_s"] = time.monotonic() - started
                    rows.append(row)
                    with (args.output / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
        # Gold labels are never loaded into the context/controller/model path.
        if args.answers:
            answers = json.loads(args.answers.read_text())
            for case in cases:
                if answers.get(case["id"]) not in [chr(65 + i) for i in range(len(case["choices"]))]:
                    raise ValueError(f"Missing or invalid scoring label for {case['id']}")
            manifest["answers_sha256"] = sha256(args.answers)
            for row in rows:
                row["correct_if_stopped_here"] = (row["answer"] == answers[row["case_id"]]) if row["complete"] else None
            (args.output / "scored_results.json").write_text(json.dumps(rows, indent=2) + "\n")
            manifest["scoring_complete"] = True
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["completed_rows"] = len(rows)
        manifest["failed_rows"] = sum(not row["complete"] for row in rows)
        manifest["status"] = (
            "complete"
            if len(rows) == manifest["planned_rows"] and not manifest["failed_rows"] and manifest["scoring_complete"]
            else "incomplete_or_failed"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if manifest["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--model-config", required=True, action="append", metavar="LABEL=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--answers", type=Path, help="Scorer-only case ID -> answer-letter map")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
