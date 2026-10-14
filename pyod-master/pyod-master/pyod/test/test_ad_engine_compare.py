# -*- coding: utf-8 -*-
"""Tests for ADEngine.compare_detectors benchmark-aware ranking (TA1).

Covers the three behavior paths:

1. Explicit `names` provided -> unchanged from pre-TA1 (returns
   explanations in input order).
2. `data_type` has a benchmark-backed ranking source in the KB:
   tabular uses ADBench `overall_top_5` (PyOD detector names);
   time_series uses each shipped detector's `benchmark_rank`
   metadata (TSB-AD overall keys). Benchmark-ranked entries come
   first, then catalog order fills `top_k`.
3. Modality without applicable benchmark ranking, or no `data_type`
   at all -> falls back to catalog order from `list_detectors`.
"""
import os
import sys
import unittest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.ad_engine import ADEngine


class TestCompareDetectorsExplicitNames(unittest.TestCase):
    """When `names` is passed, behavior is unchanged from pre-TA1."""

    def setUp(self):
        self.engine = ADEngine()

    def test_explicit_names_returned_in_input_order(self):
        names = ['ECOD', 'IForest', 'KNN']
        result = self.engine.compare_detectors(names=names)
        assert [d['name'] for d in result] == names

    def test_explicit_names_overrides_data_type_and_top_k(self):
        # Even with data_type and top_k set, explicit names win.
        result = self.engine.compare_detectors(
            names=['HBOS'], data_type='tabular', top_k=10)
        assert len(result) == 1
        assert result[0]['name'] == 'HBOS'


class TestCompareDetectorsBenchmarkRanked(unittest.TestCase):
    """TA1: tabular and time_series use benchmark-backed ranking
    sources (top-level for tabular ADBench, per-detector
    `benchmark_rank` metadata for time_series TSB-AD)."""

    def setUp(self):
        self.engine = ADEngine()

    def test_tabular_top3_matches_adbench_overall_top5_prefix(self):
        # ADBench overall_top_5 = ECOD, IForest, KNN, COPOD, HBOS
        result = self.engine.compare_detectors(
            data_type='tabular', top_k=3)
        names = [d['name'] for d in result]
        assert names == ['ECOD', 'IForest', 'KNN']

    def test_tabular_top5_matches_full_adbench_overall_top5(self):
        result = self.engine.compare_detectors(
            data_type='tabular', top_k=5)
        names = [d['name'] for d in result]
        assert names == ['ECOD', 'IForest', 'KNN', 'COPOD', 'HBOS']

    def test_tabular_topk_exceeding_ranking_appends_catalog_order(self):
        # top_k larger than 5 ranked names; expect ADBench top_5 first,
        # then remaining shipped tabular detectors in catalog order.
        result = self.engine.compare_detectors(
            data_type='tabular', top_k=10)
        names = [d['name'] for d in result]
        assert len(names) == 10
        assert names[:5] == ['ECOD', 'IForest', 'KNN', 'COPOD', 'HBOS']
        assert len(set(names)) == 10  # no duplicates

    def test_tabular_returns_only_shipped_detectors(self):
        result = self.engine.compare_detectors(
            data_type='tabular', top_k=5)
        for d in result:
            assert d.get('status') == 'shipped'

    def test_time_series_uses_tsb_ad_per_detector_ranks(self):
        # TSB-AD's overall_top_5 names do not match shipped PyOD
        # detector names, so compare_detectors must instead read each
        # detector's `benchmark_rank` metadata. Expected order
        # (lowest = best): KShape (TSB_AD_overall=2),
        # MatrixProfile (10), LSTMAD (13), SpectralResidual (14),
        # TimeSeriesOD (TSB_AD_overall_iforest=16).
        result = self.engine.compare_detectors(
            data_type='time_series', top_k=5)
        names = [d['name'] for d in result]
        catalog = [d['name'] for d in
                   self.engine.list_detectors(
                       data_type='time_series')[:5]]
        assert names == ['KShape', 'MatrixProfile', 'LSTMAD',
                         'SpectralResidual', 'TimeSeriesOD']
        assert names != catalog
        for d in result:
            assert 'time_series' in d.get('data_types', [])


class TestCompareDetectorsCatalogFallback(unittest.TestCase):
    """Modalities without an applicable benchmark ranking, and the
    no-`data_type` case, fall back to catalog order."""

    def setUp(self):
        self.engine = ADEngine()

    def test_image_falls_back_to_catalog_order(self):
        result = self.engine.compare_detectors(
            data_type='image', top_k=3)
        catalog = self.engine.list_detectors(data_type='image')
        expected = [d['name'] for d in catalog[:3]]
        assert [d['name'] for d in result] == expected

    def test_graph_falls_back_to_catalog_order(self):
        # BOND has no overall_top_5 ranking; should fall back.
        result = self.engine.compare_detectors(
            data_type='graph', top_k=3)
        catalog = self.engine.list_detectors(data_type='graph')
        expected = [d['name'] for d in catalog[:3]]
        assert [d['name'] for d in result] == expected

    def test_no_data_type_falls_back_to_catalog_order(self):
        result = self.engine.compare_detectors(top_k=3)
        catalog = self.engine.list_detectors()
        expected = [d['name'] for d in catalog[:3]]
        assert [d['name'] for d in result] == expected


if __name__ == '__main__':
    unittest.main()
