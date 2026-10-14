# -*- coding: utf-8 -*-
"""Audio encoder for EmbeddingOD: handcrafted acoustic features.

Reduces each audio clip to a fixed 74-dimensional vector: 20 MFCCs, 12
chroma bins, and 5 spectral descriptors (centroid, bandwidth, rolloff,
zero-crossing rate, RMS), each summarized by its mean and standard
deviation over frames. Any PyOD detector then runs on the resulting
tabular matrix, the model-agnostic "embed then detect" pattern applied to
audio. The feature schema follows standard acoustic descriptors computed
with ``librosa``; it needs no GPU.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np

from . import BaseEncoder

_DEFAULT_SR = 22050
_N_MFCC = 20
_MIN_SAMPLES = 2048  # pad shorter clips so the STFT has at least one frame


def _mono(y):
    """Collapse a waveform array to 1-D mono.

    1-D arrays pass through. For 2-D arrays the channel axis is the
    smaller of the two dimensions (audio has far more samples than
    channels), and it is averaged out. Higher dimensions are rejected.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        ch_axis = 0 if y.shape[0] <= y.shape[1] else 1
        return y.mean(axis=ch_axis).astype(np.float32)
    raise ValueError(
        "waveform array must be 1-D or 2-D, got %d-D" % y.ndim)


def _to_mono_waveform(item, sr):
    """Return a 1-D mono waveform for one input item.

    Accepts a file path (str), a waveform array, or a
    ``(waveform, sample_rate)`` tuple. Multi-channel arrays are averaged
    to mono. A ``(waveform, sample_rate)`` tuple is resampled to ``sr``.
    A bare array is assumed to already be at ``sr``.
    """
    import librosa

    if isinstance(item, str):
        y, _ = librosa.load(item, sr=sr, mono=True)
        return np.asarray(y, dtype=np.float32)

    if isinstance(item, tuple) and len(item) == 2:
        wav, in_sr = item
        y = _mono(wav)
        if int(in_sr) != int(sr):
            y = librosa.resample(y, orig_sr=int(in_sr), target_sr=int(sr))
        return np.asarray(y, dtype=np.float32)

    return _mono(item)


def _summarize(name, frames, out):
    """Write mean and std over frames for each row of a feature matrix."""
    means = frames.mean(axis=1)
    stds = frames.std(axis=1)
    for i in range(frames.shape[0]):
        out["%s%d_mean" % (name, i)] = float(means[i])
        out["%s%d_std" % (name, i)] = float(stds[i])


class AudioFeatureEncoder(BaseEncoder):
    """Handcrafted acoustic-feature encoder (librosa, no GPU).

    Each audio clip becomes a 74-dimensional vector: 20 MFCC, 12 chroma,
    and 5 spectral descriptors (spectral centroid, bandwidth, rolloff,
    zero-crossing rate, RMS energy), each as its mean and standard
    deviation over short-time frames at 22.05 kHz.

    Parameters
    ----------
    sr : int, optional (default=22050)
        Target sample rate. File inputs are loaded at this rate;
        ``(waveform, sample_rate)`` tuples are resampled to it.

    n_mfcc : int, optional (default=20)
        Number of MFCC coefficients.

    Notes
    -----
    Inputs may be file paths (str), mono or multi-channel waveform arrays,
    or ``(waveform, sample_rate)`` tuples. Multi-channel audio is averaged
    to mono. The output dimensionality is ``2 * (n_mfcc + 12 + 5)``; with
    the default ``n_mfcc=20`` this is 74.

    The feature extraction is deterministic given a fixed input; it does
    not depend on any random seed.

    Examples
    --------
    >>> import numpy as np
    >>> from pyod.utils.encoders.audio import AudioFeatureEncoder
    >>> clips = [np.random.RandomState(s).randn(22050) for s in range(3)]
    >>> enc = AudioFeatureEncoder()
    >>> feats = enc.encode(clips, show_progress=False)
    >>> feats.shape
    (3, 74)
    """

    def __init__(self, sr=_DEFAULT_SR, n_mfcc=_N_MFCC):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def _features_one(self, item):
        import librosa

        y = _to_mono_waveform(item, self.sr)
        if len(y) < _MIN_SAMPLES:
            y = np.pad(y, (0, _MIN_SAMPLES - len(y)))
        out = {}
        _summarize("mfcc",
                   librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc),
                   out)
        _summarize("chroma",
                   librosa.feature.chroma_stft(y=y, sr=self.sr, tuning=0.0),
                   out)
        _summarize("cent",
                   librosa.feature.spectral_centroid(y=y, sr=self.sr), out)
        _summarize("bw",
                   librosa.feature.spectral_bandwidth(y=y, sr=self.sr), out)
        _summarize("roll",
                   librosa.feature.spectral_rolloff(y=y, sr=self.sr), out)
        _summarize("zcr", librosa.feature.zero_crossing_rate(y), out)
        _summarize("rms", librosa.feature.rms(y=y), out)
        return out

    def encode(self, X, batch_size=32, show_progress=True):
        """Encode audio clips to a (n_samples, 74) feature matrix.

        Parameters
        ----------
        X : list
            Audio clips as file paths, waveform arrays, or
            ``(waveform, sample_rate)`` tuples.

        batch_size : int, optional (default=32)
            Unused; present for API consistency with other encoders.

        show_progress : bool, optional (default=True)
            Unused; present for API consistency.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, 74)
        """
        try:
            import librosa  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            raise ImportError(
                "AudioFeatureEncoder requires 'librosa' and 'soundfile'. "
                "Install with: pip install pyod[audio]")

        if len(X) == 0:
            raise ValueError("AudioFeatureEncoder received an empty input.")

        rows = [self._features_one(item) for item in X]
        # Column order is fixed by insertion order and identical across
        # clips (same feature set), so the matrix columns stay aligned.
        cols = list(rows[0].keys())
        mat = np.array([[r[c] for c in cols] for r in rows],
                       dtype=np.float64)
        return self._validate_output(mat, n_samples=len(X))
