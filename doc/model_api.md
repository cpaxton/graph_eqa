# Model API and paired context evaluation

This fork starts at upstream GraphEQA `d3fbbf14bdb05631a1633556d7d3056601374cd3`.
Planner and room inference can use any supplied model exposing the OpenAI
Chat Completions interface. No model or API key is instantiated on module import.
Credentials are read from the environment variable named by `api_key_env` and
are not included in request traces.

## Select the provider and model

Start the emet model server, or provide another compatible endpoint. Use the
model ID advertised by its `/v1/models` endpoint. For example:

```bash
# In the emet checkout, on the model host:
emet serve llm --vl --llm qwen3-vl-eqa --host 127.0.0.1 --port 8001

# In this fork, in the client environment:
export GRAPH_EQA_BASE_URL=http://127.0.0.1:8001/v1
export GRAPH_EQA_MODEL=qwen3-vl-eqa
PYTHONPATH=. python scripts/run_vlm_planner_hmeqa_habitat.py \
  --cfg_file grapheqa_hmeqa_habitat \
  --model-config cfg/models/qwen_compatible.yaml
```

**That legacy Habitat command is not a no-ground-truth benchmark.** Its semantic
sensor, dataset population filter, annotation enrichment and navmesh execution
must be replaced/qualified before making such a comparison. Model profiles only
override `vlm`; they cannot replace perception settings or the dataset. Changing
the model API does not qualify a complete upstream reproduction.

The provider is selected by `vlm.provider: openai_compatible`, independently of
the model name. Room inference uses the same model by default; an explicit
`vlm.room_model` can select another model on the same provider. Use separate
profile files with explicit endpoints/model IDs for different model servers;
reusing an environment-interpolated profile twice does not create two models.

Three response modes are explicit:

- `json_schema`: provider-enforced strict JSON Schema, followed by local validation.
- `json_object`: JSON-mode request plus an explicit schema instruction and local validation.
- `text`: schema instruction and local validation, without claiming provider-side constraints.

See the [official structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
Provider support varies. The emet prototype server uses `text`; unsupported
settings must produce an error rather than being silently ignored. There is no
automatic mode, endpoint or model fallback. `token_parameter` selects
`max_tokens` (default) or `max_completion_tokens`. Optional `temperature`, `top_p`,
`seed` and `extra_body` are sent only when configured and supported by the server.
Timeout and bounded SDK retry settings are recorded in the experiment profile.

Each output directory contains `model_calls.jsonl`: purpose, endpoint, full model
request (including image data), raw response, reported model/usage, validation
status, errors and wall time. Missing provider token counts remain unknown;
they must not be interpreted as zero-cost inference. Invalid/truncated/refused
responses fail without applying a target or replacing the model's answer.

## Paired comparisons on frozen visual evidence

`scripts/run_paired_context_eval.py` evaluates each provided model serially in
two conditions: image-only MCQ inference and the reference graph controller.
Both receive exactly the same captured PNG and question/choices, with the same
per-profile generation budget. The graph condition also receives the observed
graph, agent state and action registry. This compares complete input/controller
conditions; it does not isolate a causal graph effect or execute navigation.

Each case is a JSON object with **only** `id`, `question`, `choices`, `image`, and
`context`. Store cases as a JSON list. `image` is relative to the cases file;
use a captured current-view PNG or an already-declared reference image packet.
`context` is exactly `SceneGraphSim.planner_context()`, exported from the learned
mapper. Its action registry is preserved explicitly; the adapter never guesses
actions from node names or substitutes camera coordinates for object geometry.
Gold labels and privileged scene metadata are not accepted as case fields.

```bash
PYTHONPATH=. python scripts/run_paired_context_eval.py \
  --cases /path/to/cases.json \
  --model-config qwen8b=/path/to/qwen8b-model.yaml \
  --model-config another=/path/to/another-model.yaml \
  --output /path/to/new-output \
  --answers /path/to/scorer-only-answers.json
```

`--answers` is optional. It maps case IDs to MCQ letters and is read only after
all model calls. `correct_if_stopped_here` is a frozen-context diagnostic, not
active EQA success: a controller may correctly say it needs more observations.
No target is executed, and `interaction_steps` is zero. Infrastructure failures
remain in `results.jsonl` and the planned denominator; incomplete runs exit
nonzero. A clean committed checkout is required, including untracked source.

## Adaptations from upstream

- Provider profiles route planner and room calls through configured model APIs;
  room inference no longer silently invokes a hardcoded GPT-4o-mini. Legacy
  base configs explicitly retain `room_model: gpt-4o-mini` for historical runs;
  applying a model profile clears that dependency unless the profile explicitly
  chooses another room model. Legacy Gemini/Claude/Together planner
  implementations remain separate and are not claimed to use this adapter.
- Controller construction no longer receives a gold answer. Scoring belongs in
  the evaluator; model answers are returned without an earlier-assessment override.
- Object/room enums encode instance IDs. Upstream name-valued enums collapse
  repeated labels into aliases and cannot select the second identically named
  object. History still includes the object's human-readable label.
- PNG inputs have a matching data-URL MIME type. Leading schema/policy/question
  instructions are preserved; conversation history is not reset by the adapter.
- Disabled room inference retains generic observed room labels in the registry;
  it does not inject ground-truth room categories. Agent-state numeric formatting
  uses plain numbers rather than NumPy type wrappers.

CPU contract checks (no API credentials, Hydra or GPU required):

```bash
PYTHONPATH=. python -m pytest tests/test_model_api.py tests/test_paired_context_eval.py -q
```

Active paired evaluation additionally requires observed-space execution and a
matched, declared pose source. This frozen-context runner does not qualify them.
