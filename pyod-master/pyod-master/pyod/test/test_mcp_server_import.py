# -*- coding: utf-8 -*-
"""Import-safety tests for pyod.mcp_server.

pyod.mcp_server must be importable in a core install that does not
have the optional ``mcp`` extra. The v3.0.0 implementation called
``sys.exit(1)`` at module import time if ``mcp.server.fastmcp`` was
missing, which made ``import pyod.mcp_server`` kill any parent
process that tried to probe MCP availability.
"""
import subprocess
import sys


def test_mcp_server_imports_without_mcp_extra():
    """`import pyod.mcp_server` does not exit the process.

    Runs in a subprocess so we do not need to unload `mcp` from this
    test runner's sys.modules. The subprocess blocks `mcp` from being
    imported by shadowing it with None, then attempts to import
    pyod.mcp_server. Exit code 0 means the module was import-safe.
    """
    script = (
        "import sys\n"
        "sys.modules['mcp'] = None  # block any real mcp import\n"
        "try:\n"
        "    import pyod.mcp_server  # must not sys.exit\n"
        "    print('IMPORTED_OK')\n"
        "except SystemExit as e:\n"
        "    print(f'IMPORT_EXITED_{e.code}')\n"
        "    sys.exit(2)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"import pyod.mcp_server raised SystemExit in a core install: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "IMPORTED_OK" in result.stdout


def test_mcp_server_exposes_main():
    """pyod.mcp_server.main is callable so the unified CLI can delegate."""
    import pyod.mcp_server as m
    assert callable(getattr(m, "main", None)), (
        "pyod.mcp_server.main must be a callable so `pyod mcp serve` "
        "can delegate to it."
    )


def test_mcp_server_main_returns_nonzero_without_mcp_extra():
    """main() must return a non-zero exit code when mcp is missing, not sys.exit at import."""
    script = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"
        "import pyod.mcp_server as m\n"
        "rc = m.main()\n"
        "print(f'RC={rc}')\n"
        "sys.exit(0 if rc != 0 else 3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "RC=" in result.stdout


def test_check_mcp_handles_missing_parent_package():
    """_check_mcp() must return None (not raise) when `mcp` is not installed.

    Regression test for the Round 2 finding that
    ``importlib.util.find_spec("mcp.server.fastmcp")`` raises
    ModuleNotFoundError instead of returning None when the parent
    `mcp` package is not installed. The probe must guard against this.
    """
    script = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"
        "import pyod.mcp_server as m\n"
        "result = m._check_mcp()\n"
        "assert result is None, f'expected None, got {result!r}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"_check_mcp raised or returned non-None when mcp is missing: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_main_registers_all_ten_tools_in_order(monkeypatch):
    """Positive-path test: main() registers the 10 canonical tools in order.

    Uses a fake FastMCP class that records every callable passed through
    `mcp.tool()(fn)` and asserts the full registration sequence. Also
    verifies mcp.run() is invoked exactly once.

    The order matters: Tier-A knowledge/planning tools come first
    (profile_data, plan_detection, build_detector, list_detectors,
    explain_detector, compare_detectors, get_benchmarks), then the
    stateless Tier-B detection tools (run_detection, analyze_results,
    explain_findings).
    """
    import pyod.mcp_server as m

    registered: list[str] = []
    run_calls = {"n": 0}

    class _FakeMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def _decorator(fn):
                registered.append(fn.__name__)
                return fn
            return _decorator

        def run(self):
            run_calls["n"] += 1

    monkeypatch.setattr(m, "_check_mcp", lambda: _FakeMCP)

    rc = m.main()
    assert rc == 0
    assert run_calls["n"] == 1
    assert registered == [
        "profile_data",
        "plan_detection",
        "build_detector",
        "list_detectors",
        "explain_detector",
        "compare_detectors",
        "get_benchmarks",
        "run_detection",
        "analyze_results",
        "explain_findings",
    ]


# ----------------------------------------------------------------------
# JSON contract tests for the stateless Tier-B tools
#
# These tests call the Python tool functions directly (the same
# callables that ``main()`` would register with FastMCP), so they do
# not require the optional ``mcp`` extra. The contract under test is
# the JSON response shape that an MCP client would receive.
# ----------------------------------------------------------------------


def _write_npy(tmp_path, name, arr):
    import numpy as np
    p = tmp_path / name
    np.save(str(p), arr)
    return str(p)


def _make_data(tmp_path, n_samples=120, n_features=5, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    return _write_npy(tmp_path, 'X.npy', X)


def _plan_via_mcp(data_path):
    """Drive profile_data + plan_detection through the MCP layer."""
    import pyod.mcp_server as m
    profile = m.profile_data(data_path, data_type='tabular')
    return m.plan_detection(profile, priority='balanced')


def test_run_detection_returns_serializable_result(tmp_path):
    """run_detection JSON omits detector and lists numpy arrays."""
    import json
    import pyod.mcp_server as m
    data_path = _make_data(tmp_path)
    plan = _plan_via_mcp(data_path)
    result = m.run_detection(data_path, plan)
    out = json.loads(result)
    assert 'detector' not in out
    assert 'scores_train' in out
    assert isinstance(out['scores_train'], list)
    assert 'labels_train' in out
    assert isinstance(out['labels_train'], list)
    assert isinstance(out['threshold'], (int, float))
    assert isinstance(out['n_anomalies'], int)
    assert 'plan' in out


def test_run_detection_invalid_plan_returns_error(tmp_path):
    import json
    import pyod.mcp_server as m
    data_path = _make_data(tmp_path)
    out = json.loads(m.run_detection(data_path, "not-json"))
    assert 'error' in out


def test_analyze_results_round_trips_run_detection(tmp_path):
    """analyze_results accepts run_detection JSON and returns analysis."""
    import json
    import pyod.mcp_server as m
    data_path = _make_data(tmp_path)
    plan = _plan_via_mcp(data_path)
    result = m.run_detection(data_path, plan)
    analysis = json.loads(m.analyze_results(result, top_k=3))
    assert 'n_anomalies' in analysis
    assert 'anomaly_ratio' in analysis
    assert 'score_distribution' in analysis
    assert 'top_anomalies' in analysis
    assert len(analysis['top_anomalies']) == 3
    assert 'summary' in analysis


def test_analyze_results_invalid_result_returns_error():
    import json
    import pyod.mcp_server as m
    out = json.loads(m.analyze_results("not-json"))
    assert 'error' in out


def test_explain_findings_round_trips_run_detection(tmp_path):
    """explain_findings accepts run_detection JSON and returns rows."""
    import json
    import pyod.mcp_server as m
    data_path = _make_data(tmp_path)
    plan = _plan_via_mcp(data_path)
    result = m.run_detection(data_path, plan)
    explanations = json.loads(m.explain_findings(result, top_k=3))
    assert isinstance(explanations, list)
    assert len(explanations) == 3
    for entry in explanations:
        assert 'index' in entry
        assert 'score' in entry
        assert 'percentile' in entry
        assert 'label' in entry
        assert 'narrative' in entry


def test_explain_findings_with_explicit_indices(tmp_path):
    import json
    import pyod.mcp_server as m
    data_path = _make_data(tmp_path)
    plan = _plan_via_mcp(data_path)
    result = m.run_detection(data_path, plan)
    explanations = json.loads(
        m.explain_findings(result, indices='0,5,12'))
    indices = [e['index'] for e in explanations]
    assert indices == [0, 5, 12]


def test_explain_findings_invalid_result_returns_error():
    import json
    import pyod.mcp_server as m
    out = json.loads(m.explain_findings("not-json"))
    assert 'error' in out


def test_explain_findings_invalid_indices_returns_error(tmp_path):
    import json
    import pyod.mcp_server as m
    data_path = _make_data(tmp_path)
    plan = _plan_via_mcp(data_path)
    result = m.run_detection(data_path, plan)
    out = json.loads(m.explain_findings(result, indices='a,b,c'))
    assert 'error' in out


def test_analyze_results_malformed_arrays_returns_error():
    """Valid JSON object with non-numeric array contents must not raise."""
    import json
    import pyod.mcp_server as m
    bad = json.dumps({
        'scores_train': ['not-a-number'],
        'labels_train': [0],
        'threshold': 0.5,
        'plan': {'detector_name': 'IForest'},
    })
    out = json.loads(m.analyze_results(bad))
    assert 'error' in out


def test_explain_findings_malformed_arrays_returns_error():
    """Valid JSON object with non-numeric array contents must not raise."""
    import json
    import pyod.mcp_server as m
    bad = json.dumps({
        'scores_train': ['not-a-number'],
        'labels_train': [0],
        'threshold': 0.5,
        'plan': {'detector_name': 'IForest'},
    })
    out = json.loads(m.explain_findings(bad))
    assert 'error' in out
