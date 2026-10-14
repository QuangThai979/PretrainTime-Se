# -*- coding: utf-8 -*-
"""Tests for ADEngine.contamination_diagnostics (O5 narrowed).

The diagnostic helper must:
- require the 'analyzed' phase
- expose effective_contamination from the primary plan
- expose actual flagged_rate from the consensus labels
- expose score_percentiles at 50/75/90/95/99
- (optional) return a threshold_sweep when candidate values are passed
- never mutate state (no auto-estimation, no silent heuristic application)
"""
import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.ad_engine import ADEngine


class TestContaminationDiagnostics(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(300, 10)

    def _run_to_analyzed(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        state = self.engine.analyze(state)
        return state

    def test_requires_analyzed_phase(self):
        state = self.engine.start(self.X)
        with self.assertRaises(ValueError):
            self.engine.contamination_diagnostics(state)

    def test_returns_required_keys(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(state)
        assert set(d.keys()) >= {
            'effective_contamination', 'flagged_rate',
            'score_percentiles'}

    def test_effective_contamination_reads_from_primary_plan(self):
        # Unit-test the diagnostic's surfacing logic by injecting a
        # value directly, decoupled from whether plan_detection happens
        # to populate it (that is TA2's responsibility, fix #3).
        state = self._run_to_analyzed()
        state.plans[0].setdefault('params', {})['contamination'] = 0.15
        d = self.engine.contamination_diagnostics(state)
        assert d['effective_contamination'] == 0.15

    def test_effective_contamination_is_none_when_plan_lacks_it(self):
        state = self._run_to_analyzed()
        state.plans[0].setdefault('params', {}).pop(
            'contamination', None)
        d = self.engine.contamination_diagnostics(state)
        assert d['effective_contamination'] is None

    def test_score_percentiles_keys(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(state)
        assert sorted(d['score_percentiles'].keys()) == [
            50, 75, 90, 95, 99]

    def test_score_percentiles_monotonic(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(state)
        ps = d['score_percentiles']
        assert ps[50] <= ps[75] <= ps[90] <= ps[95] <= ps[99]

    def test_flagged_rate_matches_consensus_labels(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(state)
        labels = state.consensus['labels']
        expected = float(labels.sum()) / len(labels)
        assert abs(d['flagged_rate'] - expected) < 1e-12

    def test_threshold_sweep_returns_list_of_dicts(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(
            state, threshold_sweep=[0.05, 0.1, 0.2, 0.3])
        assert 'threshold_sweep' in d
        assert len(d['threshold_sweep']) == 4
        for entry in d['threshold_sweep']:
            assert set(entry.keys()) == {
                'contamination', 'threshold', 'flagged_rate'}

    def test_threshold_sweep_monotonic_in_flagged_rate(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(
            state, threshold_sweep=[0.05, 0.1, 0.2, 0.3])
        rates = [e['flagged_rate'] for e in d['threshold_sweep']]
        # Larger contamination => weakly larger flagged_rate.
        for a, b in zip(rates, rates[1:]):
            assert a <= b

    def test_threshold_sweep_skips_out_of_range(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(
            state, threshold_sweep=[-0.1, 0.0, 0.1, 1.0, 1.5])
        # Only c=0.1 is in (0, 1).
        assert len(d['threshold_sweep']) == 1
        assert d['threshold_sweep'][0]['contamination'] == 0.1

    def test_no_threshold_sweep_omits_key(self):
        state = self._run_to_analyzed()
        d = self.engine.contamination_diagnostics(state)
        assert 'threshold_sweep' not in d

    def test_does_not_mutate_state(self):
        state = self._run_to_analyzed()
        history_len = len(state.history)
        phase_before = state.phase
        plans_before = list(state.plans)
        self.engine.contamination_diagnostics(
            state, threshold_sweep=[0.1, 0.2])
        assert len(state.history) == history_len
        assert state.phase == phase_before
        assert state.plans == plans_before


if __name__ == '__main__':
    unittest.main()
