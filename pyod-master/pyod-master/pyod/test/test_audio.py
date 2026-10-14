# -*- coding: utf-8 -*-
"""Tests for audio anomaly detection.

Covers the handcrafted AudioFeatureEncoder and EmbeddingOD.for_audio.
All inputs are synthetic numpy waveforms, so the tests need no audio
files and no GPU. librosa is required; tests skip if it is absent.
"""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from pyod.models.embedding import EmbeddingOD
from pyod.utils.encoders import resolve_encoder
from pyod.utils.encoders.audio import AudioFeatureEncoder

SR = 22050


def _noise_clips(n, seconds=1.0, sr=SR, seed=0):
    rs = np.random.RandomState(seed)
    return [rs.randn(int(sr * seconds)).astype(np.float32) for _ in range(n)]


def _tone_clips(n, freq=440.0, seconds=1.0, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return [np.sin(2 * np.pi * freq * t).astype(np.float32) for _ in range(n)]


class TestAudioFeatureEncoder:
    def test_shape_is_74(self):
        feats = AudioFeatureEncoder().encode(_noise_clips(5),
                                             show_progress=False)
        assert feats.shape == (5, 74)
        assert np.isfinite(feats).all()

    def test_deterministic(self):
        clips = _noise_clips(4, seed=3)
        enc = AudioFeatureEncoder()
        a = enc.encode(clips, show_progress=False)
        b = enc.encode(clips, show_progress=False)
        np.testing.assert_allclose(a, b)

    def test_accepts_waveform_sr_tuple(self):
        rs = np.random.RandomState(1)
        clips = [(rs.randn(16000).astype(np.float32), 16000)
                 for _ in range(3)]
        feats = AudioFeatureEncoder().encode(clips, show_progress=False)
        assert feats.shape == (3, 74)
        assert np.isfinite(feats).all()

    def test_stereo_averaged_to_mono(self):
        rs = np.random.RandomState(2)
        mono = rs.randn(16000).astype(np.float32)
        stereo = np.stack([mono, mono])  # (2, 16000) -> averaged to mono
        f_mono = AudioFeatureEncoder().encode([mono], show_progress=False)
        f_stereo = AudioFeatureEncoder().encode([stereo], show_progress=False)
        np.testing.assert_allclose(f_mono, f_stereo, rtol=1e-5, atol=1e-5)

    def test_n_mfcc_changes_width(self):
        feats = AudioFeatureEncoder(n_mfcc=13).encode(
            _noise_clips(2), show_progress=False)
        # 2 * (13 + 12 + 5) = 60
        assert feats.shape == (2, 60)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            AudioFeatureEncoder().encode([], show_progress=False)

    def test_registry_resolves(self):
        enc = resolve_encoder('audio-mfcc')
        assert isinstance(enc, AudioFeatureEncoder)


class TestEmbeddingODAudio:
    def test_for_audio_fast_fit_predict(self):
        train = _noise_clips(20, seed=10)
        clf = EmbeddingOD.for_audio('fast', contamination=0.1,
                                    random_state=42)
        clf.fit(train)
        assert clf.decision_scores_.shape == (20,)
        assert clf.labels_.shape == (20,)
        test = _noise_clips(5, seed=11) + _tone_clips(2)
        scores = clf.decision_function(test)
        assert scores.shape == (7,)
        assert np.isfinite(scores).all()

    def test_for_audio_balanced_knn(self):
        train = _noise_clips(15, seed=20)
        clf = EmbeddingOD.for_audio('balanced')  # KNN, no torch
        clf.fit(train)
        assert clf.decision_scores_.shape == (15,)

    def test_for_audio_detects_outlier_timbre(self):
        # 18 broadband-noise clips + 2 pure tones; tones have a very
        # different spectral profile and should rank as most anomalous.
        train = _noise_clips(18, seed=30) + _tone_clips(2, freq=880.0)
        clf = EmbeddingOD.for_audio('fast', contamination=0.1,
                                    random_state=0)
        clf.fit(train)
        order = np.argsort(clf.decision_scores_)[::-1]
        top4 = set(order[:4].tolist())
        assert {18, 19} & top4  # at least one tone among the top-4

    def test_invalid_quality(self):
        with pytest.raises(ValueError):
            EmbeddingOD.for_audio('ultra')


class TestAudioAE:
    def setup_method(self):
        pytest.importorskip("torch")

    def test_fit_and_score(self):
        from pyod.models.audio_ae import AudioAE
        train = _noise_clips(16, seed=40)
        clf = AudioAE(epoch_num=3, batch_size=256, random_state=0, verbose=0)
        clf.fit(train)
        assert clf.decision_scores_.shape == (16,)
        assert clf.labels_.shape == (16,)
        assert np.isfinite(clf.decision_scores_).all()
        scores = clf.decision_function(_noise_clips(4, seed=41))
        assert scores.shape == (4,)
        assert np.isfinite(scores).all()

    def test_detects_tone_outlier(self):
        from pyod.models.audio_ae import AudioAE
        clf = AudioAE(epoch_num=15, batch_size=256, random_state=0, verbose=0)
        clf.fit(_noise_clips(20, seed=42))  # normal-only training
        test = _noise_clips(6, seed=43) + _tone_clips(3, freq=660.0)
        s = clf.decision_function(test)
        # tones (indices 6,7,8) reconstruct worse than broadband noise
        assert s[6:].mean() > s[:6].mean()

    def test_short_clip_padded(self):
        from pyod.models.audio_ae import AudioAE
        short = [np.random.RandomState(i).randn(1000).astype(np.float32)
                 for i in range(6)]
        clf = AudioAE(epoch_num=2, batch_size=128, random_state=0, verbose=0)
        clf.fit(short)
        assert clf.decision_scores_.shape == (6,)


class TestADEngineAudioRouting:
    """Audio is a first-class ADEngine modality: profiled, listed, routed."""

    def test_sniff_audio_paths(self):
        from pyod.utils.ad_engine import ADEngine
        eng = ADEngine()
        assert eng._sniff_data_type(['x.wav', 'y.flac', 'z.mp3']) == 'audio'
        assert eng._sniff_data_type(['a.txt', 'b.txt']) == 'text'
        assert eng._sniff_data_type(['p.png', 'q.jpg']) == 'image'

    def test_profile_audio(self):
        from pyod.utils.ad_engine import ADEngine
        prof = ADEngine().profile_data(['a.wav', 'b.wav', 'c.wav'])
        assert prof['data_type'] == 'audio'
        assert prof['n_samples'] == 3

    def test_list_detectors_audio(self):
        from pyod.utils.ad_engine import ADEngine
        names = {d['name'] for d in ADEngine().list_detectors(
            data_type='audio')}
        assert {'EmbeddingOD', 'AudioAE'} <= names

    def test_plan_routes_audio_to_for_audio(self):
        from pyod.utils.ad_engine import ADEngine
        plan = ADEngine().plan_detection(
            {'data_type': 'audio', 'n_samples': 100})
        assert plan['detector_name'] == 'EmbeddingOD'
        assert plan['preset'] == 'for_audio'
        # the log-mel autoencoder is offered as a deep alternative
        assert 'AudioAE' in str(plan.get('alternatives', []))
