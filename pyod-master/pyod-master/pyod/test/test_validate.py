# -*- coding: utf-8 -*-
"""Tests for ADEngine.validate hindsight evaluation (O8).

The validate helper must:
- require the 'analyzed' phase
- raise on length mismatch between y and the consensus
- raise when the consensus is missing (all detectors failed)
- compute precision/recall/F1/ROC AUC/AP for the consensus and per
  successful detector
- handle a single-class y by returning None for ROC AUC and AP
- emit FP and FN row indices that match consensus_labels vs y
- emit a consensus_vs_best diagnostic, with consensus_helped=True iff
  consensus F1 >= best-detector F1
- not mutate state
"""
import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.ad_engine import ADEngine


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        # Inject a synthetic anomaly cluster so the constructed `y`
        # has both classes present in expected proportions.
        normal = rng.randn(280, 5)
        anomaly = rng.randn(20, 5) * 3 + 5
        self.X = np.vstack([normal, anomaly])
        self.y_true = np.zeros(300, dtype=int)
        self.y_true[280:] = 1

    def _run_to_analyzed(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        state = self.engine.analyze(state)
        return state

    def test_requires_analyzed_phase(self):
        state = self.engine.start(self.X)
        with self.assertRaises(ValueError):
            self.engine.validate(state, self.y_true)

    def test_returns_required_keys(self):
        state = self._run_to_analyzed()
        v = self.engine.validate(state, self.y_true)
        assert {'consensus', 'per_detector', 'best_detector',
                'consensus_vs_best', 'false_positives',
                'false_negatives'} <= set(v.keys())

    def test_consensus_metrics_shape(self):
        state = self._run_to_analyzed()
        v = self.engine.validate(state, self.y_true)
        c = v['consensus']
        assert {'precision', 'recall', 'f1', 'roc_auc',
                'average_precision', 'n_flagged',
                'n_true_positive'} <= set(c.keys())
        assert 0.0 <= c['precision'] <= 1.0
        assert 0.0 <= c['recall'] <= 1.0
        assert 0.0 <= c['f1'] <= 1.0
        # With both classes present, ROC AUC must be a real number.
        assert c['roc_auc'] is not None
        assert c['average_precision'] is not None

    def test_per_detector_has_entry_per_success(self):
        state = self._run_to_analyzed()
        v = self.engine.validate(state, self.y_true)
        successful = [r['detector_name'] for r in state.results
                      if r.get('status') == 'success']
        for name in successful:
            assert name in v['per_detector']
            assert 'f1' in v['per_detector'][name]

    def test_best_detector_metrics_when_analysis_names_one(self):
        state = self._run_to_analyzed()
        v = self.engine.validate(state, self.y_true)
        if state.analysis and 'best_detector' in state.analysis:
            assert v['best_detector'] is not None
            assert v['consensus_vs_best']['best_detector_f1'] is not None

    def test_consensus_vs_best_helped_flag(self):
        state = self._run_to_analyzed()
        v = self.engine.validate(state, self.y_true)
        cb = v['consensus_vs_best']
        assert 'consensus_f1' in cb
        if cb['best_detector_f1'] is not None:
            assert cb['consensus_helped'] == (
                cb['consensus_f1'] >= cb['best_detector_f1'])

    def test_false_positive_negative_correctness(self):
        state = self._run_to_analyzed()
        v = self.engine.validate(state, self.y_true)
        consensus_labels = state.consensus['labels']
        for idx in v['false_positives']:
            assert consensus_labels[idx] == 1
            assert self.y_true[idx] == 0
        for idx in v['false_negatives']:
            assert consensus_labels[idx] == 0
            assert self.y_true[idx] == 1

    def test_y_length_mismatch_raises(self):
        state = self._run_to_analyzed()
        bad_y = np.zeros(len(self.X) + 5, dtype=int)
        with self.assertRaises(ValueError):
            self.engine.validate(state, bad_y)

    def test_handles_single_class_y(self):
        # All-zero y => only inlier class; ROC AUC and AP must be
        # None, not raise.
        state = self._run_to_analyzed()
        all_inlier = np.zeros(len(self.X), dtype=int)
        v = self.engine.validate(state, all_inlier)
        assert v['consensus']['roc_auc'] is None
        assert v['consensus']['average_precision'] is None

    def test_consensus_missing_raises(self):
        state = self._run_to_analyzed()
        state.consensus = None
        with self.assertRaises(ValueError):
            self.engine.validate(state, self.y_true)

    def test_does_not_mutate_state(self):
        state = self._run_to_analyzed()
        history_len = len(state.history)
        phase_before = state.phase
        self.engine.validate(state, self.y_true)
        assert len(state.history) == history_len
        assert state.phase == phase_before


if __name__ == '__main__':
    unittest.main()
