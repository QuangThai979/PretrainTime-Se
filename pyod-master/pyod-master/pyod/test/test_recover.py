"""Tests for ADEngine detector failure recovery (O3).

Covers:
- ``run()`` emits ``next_action.action='recover_detector_failure'`` on
  partial failure (some detectors failed, some succeeded).
- All-success and all-fail paths in ``run()`` are unchanged.
- ``iterate(state, {'action': 'recover'})`` substitutes failed slots,
  preserves successful plans, and resets phase to ``'planned'``.
- The recover phase guard accepts both ``'detected'`` and
  ``'analyzed'``; other actions still require ``'analyzed'``.
- ``validate_structured_feedback`` accepts ``{'action': 'recover'}``.
- ``investigation.ACTION_TYPES`` contains
  ``'recover_detector_failure'``.
"""

import unittest

import numpy as np

from pyod.utils.ad_engine import ADEngine
from pyod.utils.investigation import ACTION_TYPES


def _bogus_plan(name='__bogus_nonexistent__'):
    return {
        'detector_name': name,
        'params': {},
        'priority': 'balanced',
    }


class TestActionTypes(unittest.TestCase):
    def test_action_types_contains_recover(self):
        self.assertIn('recover_detector_failure', ACTION_TYPES)


class TestValidateRecoverFeedback(unittest.TestCase):
    def test_validate_accepts_recover_minimal(self):
        from pyod.utils._nl_feedback import validate_structured_feedback
        # Should not raise
        validate_structured_feedback({'action': 'recover'})

    def test_validate_accepts_recover_with_detectors(self):
        from pyod.utils._nl_feedback import validate_structured_feedback
        validate_structured_feedback({
            'action': 'recover', 'detectors': ['HBOS']})


class TestRunPartialFailure(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def _state_with_one_failure(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        # Force a 3-plan ensemble: keep first 2, swap the third for a
        # bogus detector that will fail at build time.
        state.plans = state.plans[:2] + [_bogus_plan()]
        return self.engine.run(state)

    def test_emits_recover_on_partial_failure(self):
        state = self._state_with_one_failure()
        self.assertEqual(state.next_action['action'],
                         'recover_detector_failure')

    def test_failed_detectors_listed(self):
        state = self._state_with_one_failure()
        self.assertEqual(state.next_action['failed_detectors'],
                         ['__bogus_nonexistent__'])

    def test_suggested_replacements_is_list(self):
        state = self._state_with_one_failure()
        self.assertIn('suggested_replacements', state.next_action)
        self.assertIsInstance(
            state.next_action['suggested_replacements'], list)

    def test_suggested_replacements_excludes_running_detectors(self):
        state = self._state_with_one_failure()
        running = {p['detector_name'] for p in state.plans
                   if p['detector_name'] != '__bogus_nonexistent__'}
        for s in state.next_action['suggested_replacements']:
            self.assertNotIn(s, running)
            self.assertNotEqual(s, '__bogus_nonexistent__')

    def test_reason_includes_failure_ratio(self):
        state = self._state_with_one_failure()
        self.assertIn('1/3', state.next_action['reason'])

    def test_consensus_uses_successful_only(self):
        state = self._state_with_one_failure()
        self.assertIsNotNone(state.consensus)
        self.assertEqual(state.consensus['n_detectors'], 2)

    def test_phase_remains_detected(self):
        state = self._state_with_one_failure()
        self.assertEqual(state.phase, 'detected')

    def test_results_record_failure(self):
        state = self._state_with_one_failure()
        statuses = [r['status'] for r in state.results]
        self.assertEqual(statuses.count('error'), 1)
        self.assertEqual(statuses.count('success'), 2)


class TestRunUnchangedPaths(unittest.TestCase):
    """Verify all-success and all-fail paths are unchanged."""

    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def test_all_success_emits_analyze(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        self.assertEqual(state.next_action['action'], 'analyze')

    def test_all_failure_emits_confirm_with_user(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state.plans = [_bogus_plan('__bogus1__'),
                       _bogus_plan('__bogus2__')]
        state = self.engine.run(state)
        self.assertEqual(state.next_action['action'],
                         'confirm_with_user')
        self.assertIsNone(state.consensus)


class TestRecoverIterate(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def _state_with_one_failure(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state.plans = state.plans[:2] + [_bogus_plan()]
        return self.engine.run(state)

    def test_recover_drops_failed_detector_name(self):
        state = self._state_with_one_failure()
        state = self.engine.iterate(state, {'action': 'recover'})
        names = [p['detector_name'] for p in state.plans]
        self.assertNotIn('__bogus_nonexistent__', names)

    def test_recover_preserves_successful_plans(self):
        state = self._state_with_one_failure()
        original_successful = [
            p['detector_name'] for p in state.plans
            if p['detector_name'] != '__bogus_nonexistent__']
        state = self.engine.iterate(state, {'action': 'recover'})
        names = [p['detector_name'] for p in state.plans]
        for n in original_successful:
            self.assertIn(n, names)

    def test_recover_resets_phase_to_planned(self):
        state = self._state_with_one_failure()
        state = self.engine.iterate(state, {'action': 'recover'})
        self.assertEqual(state.phase, 'planned')

    def test_recover_resets_detection_fields(self):
        state = self._state_with_one_failure()
        state = self.engine.iterate(state, {'action': 'recover'})
        self.assertEqual(state.results, [])
        self.assertIsNone(state.consensus)
        self.assertIsNone(state.analysis)
        self.assertIsNone(state.quality)

    def test_recover_next_action_is_run(self):
        state = self._state_with_one_failure()
        state = self.engine.iterate(state, {'action': 'recover'})
        self.assertEqual(state.next_action['action'], 'run')

    def test_recover_increments_iteration(self):
        state = self._state_with_one_failure()
        prev = state.iteration
        state = self.engine.iterate(state, {'action': 'recover'})
        self.assertEqual(state.iteration, prev + 1)

    def test_recover_with_explicit_detectors_override(self):
        state = self._state_with_one_failure()
        state = self.engine.iterate(
            state, {'action': 'recover', 'detectors': ['HBOS']})
        names = [p['detector_name'] for p in state.plans]
        self.assertIn('HBOS', names)
        self.assertNotIn('__bogus_nonexistent__', names)

    def test_recover_no_failure_is_noop(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        state = self.engine.analyze(state)
        original_plans = [p['detector_name'] for p in state.plans]
        original_results_len = len(state.results)
        state = self.engine.iterate(state, {'action': 'recover'})
        names = [p['detector_name'] for p in state.plans]
        self.assertEqual(original_plans, names)
        # No-op preserves analyzed state
        self.assertEqual(state.phase, 'analyzed')
        self.assertEqual(len(state.results), original_results_len)
        self.assertIsNotNone(state.consensus)
        self.assertIn('No failed detectors',
                      state.next_action['reason'])

    def test_recover_full_loop_run_again(self):
        """End-to-end: partial fail -> recover -> run -> analyze."""
        state = self._state_with_one_failure()
        state = self.engine.iterate(state, {'action': 'recover'})
        self.assertEqual(state.phase, 'planned')
        state = self.engine.run(state)
        self.assertIsNotNone(state.consensus)
        # Substitutes are shipped detectors and should run cleanly,
        # so run() should now reach analyze.
        self.assertEqual(state.next_action['action'], 'analyze')


class TestRecoverPhaseGuard(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def test_recover_from_detected_phase(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state.plans = state.plans[:2] + [_bogus_plan()]
        state = self.engine.run(state)
        self.assertEqual(state.phase, 'detected')
        # Should not raise
        self.engine.iterate(state, {'action': 'recover'})

    def test_recover_from_analyzed_phase(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state.plans = state.plans[:2] + [_bogus_plan()]
        state = self.engine.run(state)
        state = self.engine.analyze(state)
        self.assertEqual(state.phase, 'analyzed')
        # Should not raise
        self.engine.iterate(state, {'action': 'recover'})

    def test_recover_from_planned_phase_raises(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        with self.assertRaises(ValueError):
            self.engine.iterate(state, {'action': 'recover'})

    def test_other_actions_still_require_analyzed(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        # Phase = 'detected'; rerun should still raise
        with self.assertRaises(ValueError):
            self.engine.iterate(state, {'action': 'rerun'})


if __name__ == '__main__':
    unittest.main()
