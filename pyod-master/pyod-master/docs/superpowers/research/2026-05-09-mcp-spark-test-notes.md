# MCP Spark Test Notes: Tier-A Verified, Tier-B Gap Confirmed

**Date:** 2026-05-09
**Status:** Audit-grade evidence from a real Linux MCP run
**Purpose:** Validate the Tier-A vs Tier-B claim from
`2026-05-09-agentic-hindsight-observations.md` and the README audit, on a
host where MCP is not blocked by Windows antivirus.
**Companion file:** `2026-05-09-mcp-spark-transcript.md` (raw request /
response transcript, 7 tool calls + 7 missing-tool probes).

## Run Context

**Branch:** `development` at `c6a213d` (live mirror of GitHub upstream).

**Host:** NVIDIA DGX Spark, Ubuntu 24.04 LTS aarch64, Linux kernel
`6.17.0-1014-nvidia`. Conda env `py312` (Python 3.12.13). Reached over
SSH from a Windows daily driver via the existing `spark` alias (mDNS
`spark-37f2.local`, ed25519 key auth).

**Install:** `pip install -e ~/projects/pyod[mcp]` succeeded. Pulled in
`mcp 1.27.1`, `pydantic 2.13.4`, `starlette 1.0.0`, `httpx 0.28.1`,
`uvicorn 0.46.0` and supporting packages. Editable install resolved
`pyod 3.3.0`.

**Self-diagnostic (`pyod info`):**

```
PyOD version:          3.3.0
Detectors (ADEngine):  61 total (44 tabular, 7 time-series, 8 graph,
                       3 text, 2 image, 1 multimodal)
Classic API:           OK
ADEngine (Layer 2):    OK
MCP extra:             OK (run: pyod mcp serve)
od-expert skill:       NOT INSTALLED (Claude Code, Codex detected)
```

**Stdio smoke:** `timeout 3 python -m pyod.mcp_server </dev/null` exits
clean with code 0 (server reads EOF and shuts down). The Windows
antivirus failure mode called out in `pyod/mcp_server.py:10-12` does
not apply on this host.

**Client:** an `mcp` Python SDK stdio client (`mcp_spark_smoke.py`,
shipped to `/tmp/`) drives the server end-to-end and writes the
transcript to `/tmp/mcp-transcript.md`. Source script kept locally; the
run script is small enough to recreate verbatim from the transcript if
needed.

## Tier-A: All 7 Tools Work End-to-End

The server advertises exactly 7 tools (matches `_TOOL_FUNCTIONS` in
`pyod/mcp_server.py:247-255`):

| # | Tool | Tested input | Result |
|---|---|---|---|
| 1 | `list_detectors` | `data_type=tabular, status=shipped` | 44 detectors returned with full KB metadata (class_path, complexity, strengths, weaknesses, paper, default_params). |
| 2 | `explain_detector` | `name=ECOD` | Single-detector JSON with the same KB shape as list_detectors entries. |
| 3 | `compare_detectors` | `data_type=tabular, top_k=3` | ECOD, ABOD, COPOD. **Quirk noted below.** |
| 4 | `get_benchmarks` | `benchmark=all` | ADBench, NLP_ADBench, TSB_AD, BOND structured rankings. |
| 5 | `profile_data` | `data_path=examples/data/cardio.csv` | `tabular, n_samples=1831, n_features=21, has_nan=false, dimensionality_class=medium`. Matches the canonical cardio fingerprint in `references/workflow.md`. |
| 6 | `plan_detection` | profile from step 5, `priority=balanced` | Primary IForest with ECOD and KNN as alternatives; same triple as the cardio worked example. |
| 7 | `build_detector` | plan from step 6 | Code snippet `from pyod.models.iforest import IForest\nclf = IForest()`. |

The 5 → 6 → 7 chain (profile, plan, build) works as a clean
agent-driven flow. Verifies that the server can carry JSON state from
one call to the next when the agent threads it through.

## Tier-A Quality Issues Spotted in Transcript

These are subtle issues an MCP-only agent would hit. They were not on
the original audit list and are surfaced here for completeness:

### TA1. `compare_detectors` does not actually compare

**Evidence:** Called with `data_type=tabular, top_k=3` and got `[ECOD,
ABOD, COPOD]`. ADBench (which the same call surfaces in
`get_benchmarks`) lists the top-5 as `[ECOD, IForest, KNN, COPOD,
HBOS]`. The implementation in `ad_engine.py:1187-1208` returns the
first N from `list_detectors(data_type=...)`, which is sorted by
internal KB order, not by benchmark rank.

**Why it matters:** The tool name promises a comparison; the result is
just truncated list output. An LLM reading the description will
recommend ABOD over IForest for a generic tabular task even though
IForest is rank 2 in ADBench while ABOD is unranked.

**Fix scope:** small. Either rename the tool to reflect actual
behavior, or sort by benchmark rank when no explicit names are given.

### TA2. `plan_detection` and `build_detector` do not carry contamination

**Evidence:** Step 6 returned `params={}` for IForest. Step 7's code
snippet was `clf = IForest()` with no contamination argument. ADEngine
defaults contamination to 0.1 internally, but the MCP-only chain never
surfaces that to the agent.

**Why it matters:** The audit's O5 (contamination assumption dominates
outcome) is even more invisible through the MCP path: the agent sees
no contamination value at any step and cannot adjust it without
knowing it exists.

**Fix scope:** small. Surface the effective contamination in the plan
output, and include it in the code snippet from `build_detector`.

## Tier-B: Confirmed Missing

A probe attempted to call seven ADEngine session methods that the
audit predicted would be unreachable through MCP. All seven returned
the structured error `Unknown tool: <name>`:

```
- run_detection      → Unknown tool
- analyze_results    → Unknown tool
- explain_findings   → Unknown tool
- suggest_next_step  → Unknown tool
- investigate        → Unknown tool
- iterate            → Unknown tool
- validate           → Unknown tool
```

**Caveat on transcript labels:** the probe section in the raw
transcript labels each probe `UNEXPECTEDLY PRESENT (err=True)`. That
label comes from the client script reading FastMCP's `result.isError`
field; FastMCP returns an error-typed result for unknown tools rather
than raising. The actual response payload (`Unknown tool: X`) is
unambiguous. The labeling logic in the probe script is the bug, not
the server behavior.

**What this means concretely:**

An MCP-only LLM connecting to `python -m pyod.mcp_server` can:

- Read the detector catalog and benchmark rankings
- Profile a dataset
- Get a recommended detector and parameters
- Get a Python code snippet that constructs the detector

It cannot:

- Run the detector
- Get scores or labels back through MCP
- Run a multi-detector consensus
- Get a quality assessment, an analysis, or feature-level explanations
- Iterate on the result
- Validate against held-out labels (no `validate` exists at all,
  matches the audit's O8 finding)

The agent has to fall through to "execute the snippet locally," which
breaks the "out of the box" framing in `README.rst:125`.

## Implications for the Audit and Fix Plan

### Confirmed without change

- **README L125 mismatch is real and falsifiable.** The Linux test
  removes the Windows-AV escape clause; the gap is in the server, not
  the platform.
- **Tier-A surface is solid on Linux.** No crashes, no protocol
  errors, no missing dependencies after `pip install pyod[mcp]`. The
  MCP path is production-ready for knowledge and planning.

### New evidence not in the prior audit

- **TA1 (`compare_detectors` is a misnamed slice)** and **TA2
  (contamination invisible through MCP chain)** were surfaced by
  running, not by code reading. Both are small fixes.

### Plan adjustments to discuss

The fix plan currently defers MCP Tier-B exposure (as noted in the
plan-review checkpoint). With this evidence, two updates are worth
weighing in the upcoming Codex plan review:

1. **Reorder MCP Tier-B exposure higher**, given the Linux evidence
   removes the "maybe Windows is the problem" defense. The README
   liability is concrete.
2. **Add TA1 and TA2 as small fixes to the cycle.** Both are
   transcript-level evidence the next plan can cite directly. Each is
   smaller than the existing O6 wording fix.

These are decisions for plan-review, not unilateral changes.

## Reproducer

```bash
# On Spark (or any Linux host with miniforge / py312)
git clone -b development https://github.com/yzhao062/pyod.git
~/miniforge3/envs/py312/bin/pip install -e "./pyod[mcp]"

# Drive every advertised tool plus the 7 Tier-B probes
~/miniforge3/envs/py312/bin/python mcp_spark_smoke.py mcp-transcript.md
```

The `mcp_spark_smoke.py` source is a single asyncio script that uses
`mcp.client.stdio.stdio_client` and `ClientSession` from the official
SDK. Reproducing it is straightforward; the prior transcript file is
the canonical reference for expected output.
