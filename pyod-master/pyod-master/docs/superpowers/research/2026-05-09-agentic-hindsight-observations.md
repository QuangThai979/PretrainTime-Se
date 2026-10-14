# Agentic Hindsight Observations: UCI Ionosphere

**Date:** 2026-05-09
**Status:** Observation log from a real external-data run
**Purpose:** Use hindsight from a skill-backed agentic flow to decide how
PyOD itself should improve. This is not demo copy; it is product and API
feedback collected from one complete run.

## Run Context

**Dataset:** UCI Ionosphere, 351 radar returns, 34 numeric features.

- Dataset page: https://archive.ics.uci.edu/dataset/52/ionosphere
- Raw file used by the script:
  https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
- Labels: `g` = good radar return, `b` = bad radar return.
- Agent protocol: labels were hidden during detection and opened only for
  hindsight validation.

**Run artifact:** `examples/agentic_hindsight_real_data.py`

**Agent stack used:**

- `od-expert` packaged skill, explicitly loaded from
  `pyod.skills.get_skill_path("od-expert")`
- On-demand skill references:
  `references/workflow.md`, `references/tabular.md`
- `ADEngine` session API:
  `start -> plan -> run -> analyze -> iterate`

**Validation command:**

```powershell
C:\Users\yuezh\miniforge3\envs\py312\python.exe examples\agentic_hindsight_real_data.py
```

## Run Summary

Held-out label distribution: 126/351 bad returns, true bad-return rate
35.9%.

| Pass | Agent action | Successful detectors | Consensus flagged | Precision | Recall | F1 | ROC AUC | AP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Initial unsupervised | Default ADEngine plan | IForest, ECOD | 26 | 0.885 | 0.183 | 0.303 | 0.803 | 0.730 |
| Recovery | Exclude failed KNN, add COPOD | IForest, ECOD, COPOD | 33 | 0.909 | 0.238 | 0.377 | 0.806 | 0.724 |
| Hindsight | Set contamination to 0.359 | IForest, ECOD, COPOD | 121 | 0.612 | 0.587 | 0.599 | 0.806 | 0.724 |

Final per-detector hindsight after contamination adjustment:

| Detector | Flagged | Precision | Recall | F1 | ROC AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| IForest | 126 | 0.675 | 0.675 | 0.675 | 0.854 | 0.802 |
| ECOD | 126 | 0.540 | 0.540 | 0.540 | 0.735 | 0.657 |
| COPOD | 126 | 0.587 | 0.587 | 0.587 | 0.799 | 0.689 |

## Observations

### O1. Skill activation is still fragile across agent hosts

**Evidence:** `pyod info` reported the `od-expert` skill installed at the
Claude user-global path, while also warning that Codex does not read that
path and needs `pyod install skill --project`.

**Why it matters:** The V3 agentic story can silently fail depending on the
host. A user can have the skill installed and still not have it active in
the current agent.

**PyOD iteration:** Add a first-class, host-aware self-check to the agentic
docs and examples. For standalone scripts, prefer explicit loading through
`pyod.skills.get_skill_path()` so the example is reproducible without
depending on a host's skill-discovery rules.

### O2. "Calling a skill" has no programmatic policy interface

**Evidence:** The script could only activate the skill by reading Markdown
files and then manually implementing the checklist in Python. That is
faithful to how agent hosts inject skills, but it is not a reusable PyOD API.

**Why it matters:** If skill guidance affects behavior, PyOD needs a way to
test that behavior. Markdown-only policy is hard to benchmark against
hindsight datasets.

**PyOD iteration:** Consider generating a small machine-readable companion
for `od-expert`, for example `pyod.skills.od_expert.policy`, with helpers
such as:

- `preflight(profile, X=None) -> list[Observation]`
- `postrun_triggers(state) -> list[Trigger]`
- `recommended_iteration(state) -> dict | None`

The Markdown skill can remain the agent-facing source, but a tested policy
surface would let us run hindsight suites.

### O3. Detector failure recovery should move into ADEngine

**Evidence:** ADEngine planned `IForest`, `ECOD`, `KNN`. `KNN` failed on the
Windows py312 environment with `[WinError -1066598273] Windows Error
0xc06d007f`. ADEngine continued with 2/3 detectors; the external agent logic
had to notice the failed result, call `iterate(... exclude KNN ...)`, and add
`COPOD`.

**Why it matters:** Real agentic flows will hit optional dependency,
platform, numerical, and data-shape failures. Treating failed detectors as
ordinary partial results makes the agent look successful while the planned
ensemble degraded.

**PyOD iteration:** Add an ADEngine recovery path:

- If any planned detector fails, record a structured failure event.
- If successful detector count drops below the requested ensemble size,
  automatically re-plan or set `state.next_action` to
  `recover_detector_failure`.
- Add a fallback replacement while respecting exclusions and modality.

### O4. Planner should preflight runtime-sensitive detectors

**Evidence:** KNN remained in the plan even though it could not run in this
local environment. The error was only discovered after execution.

**Why it matters:** Agentic UX is better when predictable runtime failures
are caught before the run. This is especially important for detectors that
depend on optional libraries, compiled extensions, nearest-neighbor backends,
or GPU/torch/PyG extras.

**PyOD iteration:** Add lightweight detector availability probes. The KB
already stores status and requirements; extend it with a runtime preflight
hook that can mark detectors as unavailable for the current process and feed
that into `plan_detection(..., constraints=...)`.

### O5. The default contamination assumption dominated the outcome

**Evidence:** The first useful ensemble flagged 33/351 points (9.4%) with
precision 0.909 and recall 0.238. Hindsight labels showed the true bad-return
rate was 35.9%. Setting contamination to 0.359 changed the consensus to
121/351 flagged, precision 0.612, recall 0.587, and F1 0.599.

**Why it matters:** The ranking signal was already useful (ROC AUC about
0.806), but the threshold was wrong for the dataset. The user-facing result
looked like "few high-confidence anomalies" when the label reality was "many
bad returns."

**PyOD iteration:** Make contamination a first-class agentic uncertainty:

- Surface the default contamination in `state.next_action` and reports.
- Provide an `estimate_contamination` or threshold-sweep helper.
- When labels are later supplied, record the contamination delta as a
  hindsight finding instead of treating it as ordinary user feedback.

### O6. Current stability trigger fires as a threshold warning, not a
quality failure

**Evidence:** `state.quality['stability']` was around 0.01-0.02 on every
pass, so the skill's Trigger 4 fired. Yet the final consensus ranking had
ROC AUC 0.806, and IForest alone reached ROC AUC 0.854.

**Why it matters:** Low cutoff-gap stability is informative, but the current
skill phrasing can overstate it as "quality weak" or "noise." In this run,
low stability meant the threshold was fragile, not that the ranking was
useless.

**PyOD iteration:** Recalibrate Trigger 4 after the 2026-05 stability metric
fix:

- Treat low stability primarily as "threshold sensitivity."
- Do not conflate it with low separation.
- Consider changing the trigger threshold from a hard `< 0.5` to a tiered
  interpretation based on sample size and score distribution.

### O7. Majority-vote consensus can underperform the best detector

**Evidence:** After hindsight contamination adjustment, consensus F1 was
0.599. IForest alone achieved F1 0.675 and ROC AUC 0.854, outperforming both
the consensus labels and the other detectors.

**Why it matters:** Consensus is useful for robustness, but unweighted
majority vote can dilute a strong detector when the weaker detectors are
correlated but less accurate.

**PyOD iteration:** Add or expose weighted consensus options:

- Weight detectors by unsupervised quality diagnostics.
- Weight by agreement with rank-normalized consensus but keep a guard
  against circular self-reinforcement.
- In hindsight mode, compare consensus against the best single detector and
  record whether consensus helped or hurt.

### O8. Hindsight evaluation needs a native API

**Evidence:** Metrics were computed manually in the example with scikit-learn.
ADEngine has no `validate`, `hindsight`, or `evaluate_with_labels` method that
attaches label-based validation to the session state.

**Why it matters:** If the product loop is "run unsupervised, open labels,
learn what to improve," label-based evaluation must be a first-class
artifact. Otherwise each agent writes its own validation code and the
observations are inconsistent.

**PyOD iteration:** Add:

```python
hindsight = engine.validate(state, y)
# or
state = engine.hindsight(state, y)
```

Suggested outputs:

- precision, recall, F1, ROC AUC, average precision
- false-positive and false-negative row indices
- detector-vs-consensus comparison
- recommended next iteration, e.g. contamination calibration, detector swap,
  or supervised `XGBOD` when labels are available

### O9. Feature explanations need feature metadata

**Evidence:** `engine.explain_findings(...)` returned feature indices and
z-scores. The example had to invent `pulse_03_real` names from UCI's feature
layout.

**Why it matters:** Agent explanations are much more useful when feature
names, raw values, direction, and baseline statistics are attached. Index-only
explanations are easy to misread in real user data.

**PyOD iteration:** Extend explanation APIs to accept optional metadata:

```python
engine.explain_findings(result, X=X, feature_names=names, top_k=3)
```

Return `feature_name`, `value`, `mean` or `median`, `z_score`, and direction
where possible.

### O10. ADEngine history misses the agent's actual decisions

**Evidence:** `state.history` captured ADEngine phases: profiled, planned,
detected, analyzed, iterate. It did not capture skill-level decisions such as
"loaded tabular reference," "Trigger 2 fired," "KNN failed, refill ensemble,"
or "hindsight contamination opened."

**Why it matters:** For agentic debugging, the trace must show both tool
execution and policy decisions. Otherwise hindsight cannot distinguish
between an ADEngine limitation and an agent policy limitation.

**PyOD iteration:** Add an optional agent trace layer:

- `state.agent_trace.append(...)`
- or a generic `state.history` event type for skill/policy decisions
- include trigger id, evidence, decision, and resulting action

## Proposed Backlog

### P0: Immediate follow-ups

- Add a native hindsight validation helper to ADEngine.
- Reword/recalibrate `od-expert` Trigger 4 so low stability means threshold
  sensitivity, not necessarily bad detection.
- Add detector-failure recovery or at least a `next_action` branch that tells
  the agent to recover the ensemble.

### P1: Next iteration

- Add feature metadata support to `explain_findings`.
- Add weighted consensus or best-detector-vs-consensus diagnostics.
- Add structured skill policy helpers generated from or kept in sync with
  `od-expert` Markdown.

### P2: Hindsight harness

- Build a small benchmark harness over public labeled datasets:
  UCI Ionosphere, ODDS/Cardio, ODDS/Pima, ODDS/Letter, and one time-series
  dataset.
- Protocol: hide labels, run skill-backed ADEngine, open labels, write
  standardized observation logs.
- Use the logs to evolve ADEngine, the KB router, and the `od-expert` skill
  instead of relying on one-off demos.

## Bottom Line

The hindsight run did not say "PyOD failed." It said:

1. The ranking signal was useful.
2. Threshold policy was the dominant error source.
3. Detector failure recovery belongs in the engine or a tested policy layer.
4. The skill should be treated as executable product policy, not just
   documentation injected into an agent.
