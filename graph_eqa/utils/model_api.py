# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Configurable structured inference through an OpenAI-compatible model API.

The provider transport is independent of GraphEQA's graph, image selection and
planning policy. Each response is validated before the controller can act.
"""

import copy
import json
import math
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI, pydantic_function_tool


class ModelResponseError(RuntimeError):
    """A provider returned a refusal, incomplete response or missing content."""


def with_model_config(cfg, path):
    """Overlay only model settings; a provider profile cannot replace the dataset."""
    profile = OmegaConf.load(path)
    if set(profile) != {"vlm"} or not OmegaConf.is_dict(profile.vlm):
        raise ValueError("Model profiles must contain only a vlm mapping")
    # A provider profile owns all model roles unless it explicitly declares a
    # separate room model; do not inherit a historical GPT room-call dependency.
    if "room_model" not in profile.vlm:
        profile.vlm.room_model = None
    return OmegaConf.merge(cfg, profile)


class StructuredModelAPI:
    """One configured provider/model for typed planner and room inference.

    ``json_schema`` uses the provider's structured-output interface;
    ``json_object`` and ``text`` append an explicit schema instruction and
    validate locally. Modes never silently fall back to one another. Traces
    include exact messages (including image data URLs), responses and usage,
    but never authentication headers or resolved API keys.
    """

    def __init__(self, cfg, output_path=None, *, purpose="planner", client=None, model=None):
        cfg = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        self.model = model or cfg["name"]
        if not self.model:
            raise ValueError("An explicit model name is required")
        self.mode = cfg.get("response_mode", "json_schema")
        if self.mode not in {"json_schema", "json_object", "text"}:
            raise ValueError(f"Unknown response_mode: {self.mode}")
        timeout, retries = float(cfg.get("timeout_s", 120)), int(cfg.get("max_retries", 0))
        if not math.isfinite(timeout) or timeout <= 0 or not 0 <= retries <= 5:
            raise ValueError("Require positive timeout_s and max_retries in [0, 5]")
        self.generation = {}
        token_parameter = cfg.get("token_parameter", "max_tokens")
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Invalid token_parameter")
        if cfg.get("max_tokens") is not None:
            if int(cfg["max_tokens"]) <= 0:
                raise ValueError("max_tokens must be positive")
            self.generation[token_parameter] = int(cfg["max_tokens"])
        for key in ("temperature", "top_p", "seed", "extra_body"):
            if cfg.get(key) is not None:
                self.generation[key] = cfg[key]
        base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
        # Unauthenticated local providers still require an SDK placeholder.
        # Public OpenAI retains its normal missing-key error when no URL is set.
        self.client = client or OpenAI(
            base_url=base_url, api_key=api_key or ("unused" if base_url else None), timeout=timeout, max_retries=retries
        )
        self.purpose = purpose
        self.output = Path(output_path) if output_path is not None else None

    def complete(self, messages, response_type):
        """Return the validated Pydantic response; failures propagate to callers."""
        request = {"model": self.model, "messages": copy.deepcopy(messages), **self.generation}
        schema = pydantic_function_tool(response_type)["function"]["parameters"]
        if self.mode == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_type.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            # Put the root response shape before nested definitions. Otherwise
            # text-only providers may emit an Answer definition as the entire
            # response instead of the planner object containing it.
            root_schema = {
                key: schema[key] for key in ("type", "properties", "required", "additionalProperties") if key in schema
            }
            root_schema.update(
                {key: value for key, value in schema.items() if key not in root_schema and key != "$defs"}
            )
            if "$defs" in schema:
                root_schema["$defs"] = schema["$defs"]
            request["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": "Return one complete JSON object matching the ROOT response schema below. "
                    "Definitions in $defs describe nested values, not a separate response. "
                    "Return only JSON, without surrounding prose.\n" + json.dumps(root_schema),
                },
            )
            if self.mode == "json_object":
                request["response_format"] = {"type": "json_object"}
        record = {
            "purpose": self.purpose,
            "base_url": str(self.client.base_url),
            "response_mode": self.mode,
            "request": request,
        }
        started = time.monotonic()
        try:
            completion = self.client.chat.completions.create(**request)
            record["response"] = completion.model_dump(mode="json")
            if not completion.choices:
                raise ModelResponseError("Provider returned no choices")
            choice = completion.choices[0]
            if choice.message.refusal:
                raise ModelResponseError(f"Model refusal: {choice.message.refusal}")
            if choice.finish_reason != "stop":
                raise ModelResponseError(f"Incomplete response: finish_reason={choice.finish_reason}")
            if not choice.message.content:
                raise ModelResponseError("Provider returned no JSON content")
            parsed = response_type.model_validate_json(choice.message.content)
            record["validated"] = True
            return parsed
        except Exception as exc:
            record.update(validated=False, error=f"{type(exc).__name__}: {exc}")
            response = getattr(exc, "response", None)
            if response is not None:
                record.update(http_status=response.status_code, provider_error=response.text)
            raise
        finally:
            record["wall_s"] = time.monotonic() - started
            if self.output is not None:
                self.output.mkdir(parents=True, exist_ok=True)
                with (self.output / "model_calls.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
