# -*- coding: utf-8 -*-


import os
import sys
import unittest
from platform import python_version

# noinspection PyProtectedMember
from numpy.testing import (assert_allclose, assert_array_less, assert_equal,
                           assert_raises)
from packaging.version import Version
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from pyod.models.kde import KDE
from pyod.utils.data import generate_data

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

global py_ver
py_ver = Version(python_version()) > Version('3.6.15')


def _probe_pythresh_resources():
    """Probe whether pythresh can load its bundled `.pkl` model resources.

    pythresh's META threshold loads pickled sklearn models from
    ``pythresh/models/`` via ``importlib.resources.files()``. On
    Python 3.9 with pythresh wheels that ship ``pythresh/models/``
    as a namespace package (no ``__init__.py``), the legacy
    ``files()`` implementation does ``pathlib.Path(spec.origin).parent``
    and raises ``TypeError`` because ``spec.origin`` is ``None``.
    pythresh 1.0.3 ships an ``__init__.py``; 1.1.0 does not. On
    Python 3.12+ the modern ``files()`` handles the namespace case,
    so the bug only manifests on Python 3.9 with broken pythresh
    wheels.

    Returns the original ``TypeError`` when the resource path is
    broken so the caller can include the diagnostic in a skip
    message; returns ``None`` when pythresh is healthy.
    """
    try:
        import numpy as np
        from pythresh.thresholds.meta import META
        # META.eval() exercises the exact files() call path that
        # breaks. Three points are enough; META does not require a
        # specific score distribution to reach the resource load.
        META().eval(np.array([0.1, 0.5, 0.9]))
    except TypeError as exc:
        msg = str(exc)
        # The specific bug always reports both terms because Python's
        # `pathlib.Path(None)` raises with "expected str, bytes or
        # os.PathLike object, not NoneType". Requiring both keeps the
        # probe pinned to that fingerprint and re-raises any unrelated
        # TypeError so a real pythresh or pyod regression is not masked.
        if 'PathLike' in msg and 'NoneType' in msg:
            return exc
        raise
    except Exception:
        # Any other exception (ImportError, etc.) means pythresh
        # has a different problem the caller should not paper over.
        raise
    return None


class TestThresholds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe_error = _probe_pythresh_resources()
        if probe_error is not None:
            raise unittest.SkipTest(
                "pythresh's META threshold cannot load its bundled "
                "model resources in this environment "
                "(importlib.resources.files('pythresh.models') fails "
                "because pythresh.models is a namespace package and "
                "Python 3.9's legacy files() requires "
                "spec.origin). pythresh 1.1.0 wheels lack the "
                "pythresh/models/__init__.py shipped in 1.0.3. "
                "Re-running the suite once pythresh ships a wheel "
                "with the missing __init__.py will exercise these "
                "tests again. Probe error: %s" % probe_error)

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def setUp(self):
        from pyod.models.thresholds import (AUCP, BOOT, CHAU, CLF, CLUST,
                                            CPD, DECOMP, DSN, EB, FGD, FILTER,
                                            FWFM, GAMGMM, GESD, HIST, IQR, KARCH, 
                                            MAD, MCST, META, MIXMOD, MOLL, MTT, 
                                            OCSVM, QMCD, REGR, VAE, WIND, YJ, ZSCORE)

        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train,
            n_test=self.n_test,
            contamination=self.contamination,
            random_state=42,
        )

        self.contam = [AUCP(), BOOT(), CHAU(), CLF(), CLUST(), CPD(), 
                       DECOMP(), DSN(), EB(), FGD(), FILTER(), FWFM(), 
                       GAMGMM(skip=True), GESD(), HIST(), IQR(), KARCH(), 
                       MAD(), MCST(), META(), MIXMOD(), MOLL(), MTT(), 
                       OCSVM(), QMCD(), REGR(), VAE(), WIND(), YJ(), ZSCORE()]

        for contam in self.contam:
            self.clf = KDE(contamination=contam)
            self.clf.fit(self.X_train)

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_parameters(self):
        assert (
                hasattr(self.clf, "decision_scores_")
                and self.clf.decision_scores_ is not None
        )
        assert hasattr(self.clf, "labels_") and self.clf.labels_ is not None
        assert hasattr(self.clf,
                       "threshold_") and self.clf.threshold_ is not None
        assert hasattr(self.clf, "_mu") and self.clf._mu is not None
        assert hasattr(self.clf, "_sigma") and self.clf._sigma is not None

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert roc_auc_score(self.y_test, pred_scores) >= self.roc_floor

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method="linear")
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method="unify")
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method="something")

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(
            self.X_test, method="linear", return_confidence=True
        )
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring="roc_auc_score")
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring="prc_n_score")
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring="something")

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_predict_rank(self):
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=4)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_predict_rank_normalized(self):
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=4)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    @unittest.skipIf(not py_ver, 'Python 3.6 not included')
    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()