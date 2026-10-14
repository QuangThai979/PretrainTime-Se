# -*- coding: utf-8 -*-
"""Tests for pyod 3.5.3 Surface 1 (KB-tools for agent-driven routing).

Covers:
- ADEngine.get_kb_for_routing(profile, top_k, constraints)
- ADEngine.make_plan(detector_choices, justifications, params)
- ADEngine.plan_detection(..., top_k=K) parameter generalization
"""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.ad_engine import ADEngine


class TestGetKbForRouting(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine(random_state=42)
        rng = np.random.RandomState(0)
        self.X = rng.normal(size=(200, 8))
        self.X[-10:] += 4 * rng.normal(size=(10, 8))
        self.profile = self.engine.profile_data(self.X, data_type='tabular')

    def test_returns_expected_keys(self):
        kb = self.engine.get_kb_for_routing(self.profile, top_k=3)
        for k in ('task_profile', 'available_detectors',
                  'top_k_requested', 'response_format_hint',
                  'n_available'):
            assert k in kb, f"missing key {k}"

    def test_top_k_field_preserved(self):
        kb = self.engine.get_kb_for_routing(self.profile, top_k=5)
        assert kb['top_k_requested'] == 5
        # Non-positive top_k is clamped to 1
        kb1 = self.engine.get_kb_for_routing(self.profile, top_k=0)
        assert kb1['top_k_requested'] == 1

    def test_detector_entries_have_kb_fields(self):
        kb = self.engine.get_kb_for_routing(self.profile)
        assert kb['n_available'] > 0
        sample = kb['available_detectors'][0]
        for k in ('name', 'category', 'complexity_time', 'complexity_space',
                  'strengths', 'weaknesses', 'best_for', 'avoid_when',
                  'benchmark_rank', 'modality_match'):
            assert k in sample, f"missing detector field {k}"
        assert isinstance(sample['strengths'], list)
        assert isinstance(sample['weaknesses'], list)

    def test_exclude_constraint(self):
        kb = self.engine.get_kb_for_routing(
            self.profile, constraints={'exclude_detectors': ['IForest', 'KNN']})
        names = [d['name'] for d in kb['available_detectors']]
        assert 'IForest' not in names
        assert 'KNN' not in names

    def test_data_type_strict_filter(self):
        # Default data_type_strict=True drops detectors whose data_types do
        # not include the profile's modality.
        kb_strict = self.engine.get_kb_for_routing(self.profile)
        kb_lax = self.engine.get_kb_for_routing(
            self.profile, constraints={'data_type_strict': False})
        assert kb_strict['n_available'] <= kb_lax['n_available']

    def test_pure_function(self):
        # No state mutation; profile dict is not modified.
        before = dict(self.profile)
        self.engine.get_kb_for_routing(self.profile, top_k=3)
        assert self.profile == before

    def test_bad_profile_raises(self):
        with self.assertRaises(ValueError):
            self.engine.get_kb_for_routing("not a dict")

    def test_ranking_order_iforest_first_on_tabular(self):
        # KB benchmark_rank should place IForest near the top on ADBench.
        kb = self.engine.get_kb_for_routing(self.profile)
        names = [d['name'] for d in kb['available_detectors'][:5]]
        assert 'IForest' in names

    def test_ranking_uses_modality_specific_keys_time_series(self):
        """Med-2 (Codex Round 1): time_series should sort by TSB_AD_overall.

        Asserts the modality-specific rank key is consulted by checking
        that the available_detectors list is sorted by TSB_AD_overall
        (ascending) for any pair of detectors that both declare it.
        """
        ts_profile = dict(self.profile)
        ts_profile['data_type'] = 'time_series'
        kb = self.engine.get_kb_for_routing(ts_profile)
        ts_ranked = [
            (d['name'], (d.get('benchmark_rank') or {}).get('TSB_AD_overall'))
            for d in kb['available_detectors']
            if (d.get('benchmark_rank') or {}).get('TSB_AD_overall') is not None
        ]
        # Any pair with TSB_AD_overall must be in non-decreasing rank order.
        ranks = [r for _, r in ts_ranked]
        assert ranks == sorted(ranks), (
            f"time_series ordering ignored TSB_AD_overall: {ts_ranked}")

    def test_ranking_uses_modality_specific_keys_graph(self):
        """Med-2 (Codex Round 1): graph should sort by BOND_deep / BOND_overall."""
        g_profile = dict(self.profile)
        g_profile['data_type'] = 'graph'
        kb = self.engine.get_kb_for_routing(g_profile)
        g_ranked = []
        for d in kb['available_detectors']:
            br = d.get('benchmark_rank') or {}
            r = br.get('BOND_deep') or br.get('BOND_overall')
            if r is not None:
                g_ranked.append((d['name'], r))
        ranks = [r for _, r in g_ranked]
        assert ranks == sorted(ranks), (
            f"graph ordering ignored BOND keys: {g_ranked}")


class TestMakePlan(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine(random_state=42)

    def test_single_detector(self):
        plan = self.engine.make_plan(['IForest'])
        assert plan['detector_name'] == 'IForest'
        assert plan['confidence'] == 0.7
        assert plan['alternatives'] == []

    def test_three_detectors_with_justifications(self):
        plan = self.engine.make_plan(
            ['IForest', 'KNN', 'ECOD'],
            justifications=['rank', 'proximity', 'parameter-free'])
        assert plan['detector_name'] == 'IForest'
        assert plan['reason'] == 'rank'
        assert len(plan['alternatives']) == 2
        assert plan['alternatives'][0]['detector_name'] == 'KNN'
        assert plan['alternatives'][0]['confidence'] == 0.5
        assert plan['alternatives'][1]['detector_name'] == 'ECOD'

    def test_default_justification(self):
        plan = self.engine.make_plan(['IForest', 'KNN'])
        assert 'caller-selected' in plan['reason']
        assert 'caller-selected' in plan['alternatives'][0]['reason']

    def test_unknown_detector_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.engine.make_plan(['DoesNotExist'])
        assert 'DoesNotExist' in str(cm.exception)
        assert 'case-sensitive' in str(cm.exception)

    def test_empty_choices_raises(self):
        with self.assertRaises(ValueError):
            self.engine.make_plan([])

    def test_non_list_raises(self):
        with self.assertRaises(ValueError):
            self.engine.make_plan('IForest')

    def test_params_overlay_keeps_contamination(self):
        plan = self.engine.make_plan(
            ['IForest'],
            params=[{'n_estimators': 50}])
        assert plan['params'].get('n_estimators') == 50
        # _with_contamination should add contamination if KB has a default
        # (it does for IForest); we check it's present, not the exact value.
        assert 'contamination' in plan['params']

    def test_build_detector_consumes_plan(self):
        # The plan returned by make_plan must be consumable by build_detector
        # so existing detector-construction code keeps working.
        plan = self.engine.make_plan(['IForest'])
        detector = self.engine.build_detector(plan)
        assert detector is not None
        # And it should be fittable
        rng = np.random.RandomState(7)
        X = rng.normal(size=(100, 5))
        detector.fit(X)
        assert hasattr(detector, 'decision_scores_')


class TestPlanDetectionSignature(unittest.TestCase):
    """Med-1 (Codex Round 2): new params must be keyword-only."""

    def test_new_params_are_keyword_only(self):
        import inspect
        sig = inspect.signature(ADEngine.plan_detection)
        for name in ('top_k', 'llm_client', 'llm_strict'):
            kind = sig.parameters[name].kind
            assert kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name!r} should be KEYWORD_ONLY, got {kind!r}; "
                "CHANGES.txt advertises keyword-only and callers must "
                "not bind these positionally.")
        # The v3.5.2 params remain positional-or-keyword for backward compat.
        for name in ('profile', 'priority', 'constraints'):
            kind = sig.parameters[name].kind
            assert kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
                f"v3.5.2 param {name!r} must stay POSITIONAL_OR_KEYWORD; "
                f"got {kind!r}")


class TestPlanDetectionTopK(unittest.TestCase):
    """plan_detection(..., top_k=K) generalizes the v3.5.2 valid[1:3] cap."""

    def setUp(self):
        self.engine = ADEngine(random_state=42)
        rng = np.random.RandomState(0)
        self.X = rng.normal(size=(300, 8))
        self.profile = self.engine.profile_data(self.X, data_type='tabular')

    def test_default_top_k_matches_v352_behavior(self):
        # v3.5.2 used valid[1:3] which gave up to 2 alternatives.
        plan = self.engine.plan_detection(self.profile)
        assert len(plan.get('alternatives', [])) <= 2

    def test_top_k_1(self):
        plan = self.engine.plan_detection(self.profile, top_k=1)
        assert plan.get('alternatives', []) == []

    def test_top_k_5(self):
        plan = self.engine.plan_detection(self.profile, top_k=5)
        # plan returns primary + up to top_k-1 alternatives.
        assert len(plan.get('alternatives', [])) <= 4

    def test_top_k_clamped_to_1(self):
        plan_neg = self.engine.plan_detection(self.profile, top_k=-1)
        plan_zero = self.engine.plan_detection(self.profile, top_k=0)
        plan_one = self.engine.plan_detection(self.profile, top_k=1)
        # All three should yield the same single-primary plan (no alternatives)
        assert plan_neg.get('alternatives') == plan_one.get('alternatives')
        assert plan_zero.get('alternatives') == plan_one.get('alternatives')


class TestSurface2LlmClient(unittest.TestCase):
    """plan_detection(llm_client=...) Surface 2 routing."""

    def setUp(self):
        self.engine = ADEngine(random_state=42)
        rng = np.random.RandomState(0)
        self.X = rng.normal(size=(200, 8))
        self.profile = self.engine.profile_data(self.X, data_type='tabular')

    def test_stub_llm_returns_canned_plan(self):
        def stub_llm(prompt: str) -> str:
            return ('[{"detector":"IForest","justification":"top rank"},'
                    '{"detector":"ECOD","justification":"parameter-free"},'
                    '{"detector":"KNN","justification":"proximity"}]')
        plan = self.engine.plan_detection(self.profile, llm_client=stub_llm)
        assert plan['detector_name'] == 'IForest'
        assert plan.get('note') == 'llm-driven via plan_detection(llm_client=...)'
        assert plan.get('evidence') == ['llm_routing']
        alts = [a['detector_name'] for a in plan.get('alternatives', [])]
        assert alts == ['ECOD', 'KNN']

    def test_top_k_truncates_llm_response(self):
        def stub_llm(prompt: str) -> str:
            return ('[{"detector":"IForest","justification":"a"},'
                    '{"detector":"ECOD","justification":"b"},'
                    '{"detector":"KNN","justification":"c"},'
                    '{"detector":"LOF","justification":"d"},'
                    '{"detector":"HBOS","justification":"e"}]')
        plan = self.engine.plan_detection(self.profile, llm_client=stub_llm,
                                          top_k=2)
        assert plan['detector_name'] == 'IForest'
        assert len(plan.get('alternatives', [])) == 1

    def test_llm_response_cannot_select_excluded_detector(self):
        """High-1 (Codex Round 1): LLM output must not bypass exclude_detectors."""
        import warnings

        def excluded_llm(prompt: str) -> str:
            return '[{"detector":"IForest","justification":"trying to bypass"}]'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = self.engine.plan_detection(
                self.profile,
                constraints={'exclude_detectors': ['IForest']},
                llm_client=excluded_llm)
        # Must have fallen back to rule routing.
        assert plan['detector_name'] != 'IForest'
        assert plan.get('evidence') != ['llm_routing']
        # And a RuntimeWarning must have explained the fallback.
        assert any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_per_call_llm_strict_true_reraises(self):
        """Med-3 (Codex Round 1): explicit llm_strict=True re-raises."""
        from pyod.utils._llm import RoutingParseError
        def bad_llm(prompt: str) -> str:
            return "I cannot help with that."
        with self.assertRaises(RoutingParseError):
            self.engine.plan_detection(self.profile,
                                        llm_client=bad_llm,
                                        llm_strict=True)

    def test_per_call_llm_strict_false_overrides_env(self):
        """Med-3 (Codex Round 1): explicit llm_strict=False overrides env var."""
        import os, warnings
        os.environ['PYOD3_LLM_STRICT'] = '1'
        try:
            def bad_llm(prompt: str) -> str:
                return "no json"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                plan = self.engine.plan_detection(
                    self.profile, llm_client=bad_llm, llm_strict=False)
            # Did not raise -- explicit kwarg won over env var.
            assert plan['detector_name']
            assert any(issubclass(x.category, RuntimeWarning) for x in w)
        finally:
            del os.environ['PYOD3_LLM_STRICT']

    def test_per_call_llm_strict_none_defers_to_env(self):
        """Med-3 (Codex Round 1): llm_strict=None reads PYOD3_LLM_STRICT."""
        import os
        from pyod.utils._llm import RoutingParseError
        os.environ['PYOD3_LLM_STRICT'] = '1'
        try:
            def bad_llm(prompt: str) -> str:
                return "no json"
            with self.assertRaises(RoutingParseError):
                self.engine.plan_detection(self.profile,
                                           llm_client=bad_llm,
                                           llm_strict=None)
        finally:
            del os.environ['PYOD3_LLM_STRICT']

    def test_malformed_response_falls_back_to_rules(self):
        import warnings
        def bad_llm(prompt: str) -> str:
            return "Sorry, I cannot help with that."
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = self.engine.plan_detection(self.profile,
                                              llm_client=bad_llm)
        assert plan['detector_name']  # rule-driven primary
        assert plan.get('evidence') != ['llm_routing']
        # A RuntimeWarning should have been emitted
        runtime_warnings = [x for x in w
                            if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1

    def test_strict_mode_reraises(self):
        import os
        from pyod.utils._llm import RoutingParseError
        def bad_llm(prompt: str) -> str:
            return "no JSON here"
        os.environ['PYOD3_LLM_STRICT'] = '1'
        try:
            with self.assertRaises(RoutingParseError):
                self.engine.plan_detection(self.profile, llm_client=bad_llm)
        finally:
            del os.environ['PYOD3_LLM_STRICT']

    def test_llm_client_none_preserves_rule_routing(self):
        # llm_client=None must produce the exact v3.5.2 rule plan.
        plan_default = self.engine.plan_detection(self.profile)
        plan_none = self.engine.plan_detection(self.profile, llm_client=None)
        assert plan_default == plan_none


class TestPromptBuilderModalityRank(unittest.TestCase):
    """Med-2 (Codex Round 2): the LLM prompt must carry the modality-
    specific benchmark rank, not just the modality-title-overall key.
    """

    def setUp(self):
        self.engine = ADEngine(random_state=42)
        rng = np.random.RandomState(0)
        self.X = rng.normal(size=(200, 8))

    def _profile_for(self, data_type: str) -> dict:
        # profile_data only inspects X; data_type just rides through.
        prof = self.engine.profile_data(self.X, data_type=data_type)
        return prof

    def test_text_modality_prompt_falls_back_when_no_rank_data(self):
        # text detectors currently ship without benchmark_rank entries
        # in the KB. The prompt must still render successfully (no rank
        # annotation, no crash) and must NOT use the old broken
        # `Text_overall` key name (which never existed in the KB).
        from pyod.utils._llm import build_routing_prompt
        prof = self._profile_for('text')
        kb = self.engine.get_kb_for_routing(prof)
        prompt = build_routing_prompt(kb, top_k=3)
        assert 'TASK PROFILE' in prompt
        assert 'Text_overall' not in prompt, (
            "build_routing_prompt should not emit the legacy "
            "`Text_overall` key form -- it was never a real KB field")

    def test_time_series_modality_prompt_shows_tsb_ad_rank(self):
        from pyod.utils._llm import build_routing_prompt
        prof = self._profile_for('time_series')
        kb = self.engine.get_kb_for_routing(prof)
        prompt = build_routing_prompt(kb, top_k=3)
        # TSB_AD_overall is the documented primary key for time_series.
        assert 'TSB_AD_overall' in prompt, (
            "time_series prompt should annotate the rank with TSB_AD_overall")

    def test_kb_entries_stamp_resolved_rank(self):
        # The contract: get_kb_for_routing places `resolved_rank` and
        # `resolved_rank_key` on each detector entry so downstream tools
        # do not have to re-do the lookup.
        prof = self._profile_for('time_series')
        kb = self.engine.get_kb_for_routing(prof)
        any_ranked = [d for d in kb['available_detectors']
                      if d.get('resolved_rank') is not None]
        assert any_ranked, (
            "time_series KB should expose resolved_rank on at least "
            "one entry")
        for d in any_ranked:
            assert d.get('resolved_rank_key') is not None, (
                f"detector {d['name']} has resolved_rank but no "
                "resolved_rank_key -- contract violation")


class TestRoutingResponseParser(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine(random_state=42)
        self.kb = self.engine.kb

    def test_parse_plain_json_array(self):
        from pyod.utils._llm import parse_routing_response
        resp = '[{"detector":"IForest","justification":"x"}]'
        choices, justs = parse_routing_response(resp, self.kb, top_k=3)
        assert choices == ['IForest']
        assert justs == ['x']

    def test_parse_tolerates_prose(self):
        from pyod.utils._llm import parse_routing_response
        resp = ('I recommend the following three detectors:\n'
                '[{"detector":"IForest","justification":"x"},'
                '{"detector":"ECOD","justification":"y"}]\n'
                'Hope this helps.')
        choices, _ = parse_routing_response(resp, self.kb, top_k=3)
        assert choices == ['IForest', 'ECOD']

    def test_parse_tolerates_markdown_fences(self):
        from pyod.utils._llm import parse_routing_response
        resp = ('```json\n'
                '[{"detector":"IForest","justification":"x"}]\n'
                '```')
        choices, _ = parse_routing_response(resp, self.kb, top_k=3)
        assert choices == ['IForest']

    def test_parse_skips_unknown_detectors(self):
        from pyod.utils._llm import parse_routing_response
        resp = ('[{"detector":"BogusDetector","justification":"x"},'
                '{"detector":"IForest","justification":"y"}]')
        choices, _ = parse_routing_response(resp, self.kb, top_k=3)
        assert choices == ['IForest']

    def test_parse_dedupes(self):
        from pyod.utils._llm import parse_routing_response
        resp = ('[{"detector":"IForest","justification":"a"},'
                '{"detector":"IForest","justification":"b"}]')
        choices, _ = parse_routing_response(resp, self.kb, top_k=3)
        assert choices == ['IForest']

    def test_parse_truncates_to_top_k(self):
        from pyod.utils._llm import parse_routing_response
        resp = ('[{"detector":"IForest"},'
                '{"detector":"ECOD"},'
                '{"detector":"KNN"},'
                '{"detector":"LOF"}]')
        choices, _ = parse_routing_response(resp, self.kb, top_k=2)
        assert choices == ['IForest', 'ECOD']

    def test_parse_raises_on_no_array(self):
        from pyod.utils._llm import parse_routing_response, RoutingParseError
        with self.assertRaises(RoutingParseError):
            parse_routing_response("no json here", self.kb)

    def test_parse_raises_on_all_invalid(self):
        from pyod.utils._llm import parse_routing_response, RoutingParseError
        resp = '[{"detector":"BogusOne"},{"detector":"BogusTwo"}]'
        with self.assertRaises(RoutingParseError):
            parse_routing_response(resp, self.kb)

    def test_parse_accepts_bare_string_entries(self):
        from pyod.utils._llm import parse_routing_response
        resp = '["IForest", "ECOD"]'
        choices, justs = parse_routing_response(resp, self.kb, top_k=3)
        assert choices == ['IForest', 'ECOD']
        assert justs == ['', '']


if __name__ == "__main__":
    unittest.main()
