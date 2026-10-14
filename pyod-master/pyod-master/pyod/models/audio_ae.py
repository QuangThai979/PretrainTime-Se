# -*- coding: utf-8 -*-
"""AudioAE: a log-mel reconstruction autoencoder for audio anomaly detection.

Each clip is turned into overlapping log-mel context windows; a dense
autoencoder is fit on the windows of the (mostly normal) training clips,
and each clip is scored by its mean per-window reconstruction error. This
is the DCASE-style audio anomaly detection baseline, expressed through
PyOD's ``AutoEncoder`` so the training loop and preprocessing are shared
with the rest of the library.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ..utils.encoders.audio import _to_mono_waveform

_DEFAULT_SR = 22050


def _logmel_windows(y, sr, n_mels, context, hop_length):
    """Return overlapping log-mel context windows for one waveform.

    Output shape is ``(n_windows, n_mels * context)``. Clips shorter than
    one context window are padded so at least one window is produced.
    """
    import librosa

    spec = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length))
    n_frames = spec.shape[1]
    if n_frames < context:
        pad = np.zeros((n_mels, context - n_frames), dtype=spec.dtype)
        spec = np.concatenate([spec, pad], axis=1)
        n_frames = context
    windows = [spec[:, t:t + context].T.reshape(-1)
               for t in range(n_frames - context + 1)]
    return np.stack(windows).astype(np.float32)


class AudioAE(BaseDetector):
    """Log-mel reconstruction autoencoder for audio anomaly detection.

    The detector extracts overlapping log-mel context windows from each
    clip, fits a dense autoencoder (PyOD's :class:`AutoEncoder`) on the
    windows of the training clips, and scores each clip by its mean
    per-window reconstruction error. Training assumes the input is mostly
    normal, the usual unsupervised setting.

    Requires ``torch`` (for the autoencoder) and ``pyod[audio]``
    (``librosa``, ``soundfile``).

    Parameters
    ----------
    n_mels : int, optional (default=64)
        Number of mel bands in the spectrogram.

    context : int, optional (default=5)
        Number of consecutive frames stacked into one autoencoder input
        window. The window dimensionality is ``n_mels * context``.

    hop_length : int, optional (default=512)
        STFT hop length in samples.

    sr : int, optional (default=22050)
        Target sample rate. File inputs are loaded at this rate;
        ``(waveform, sample_rate)`` tuples are resampled to it.

    contamination : float, optional (default=0.1)
        Expected proportion of outliers, used for the clip-level
        threshold and labels.

    epoch_num : int, optional (default=40)
        Autoencoder training epochs.

    batch_size : int, optional (default=1024)
        Autoencoder mini-batch size (over frames, not clips).

    lr : float, optional (default=1e-3)
        Learning rate.

    hidden_neuron_list : list of int or None, optional (default=None)
        Encoder hidden sizes. ``None`` uses ``[128, 32, 8]``, which gives
        the DCASE-style 320-128-32-8 contraction for the default
        320-dimensional window (``n_mels=64``, ``context=5``).

    device : str or None, optional (default=None)
        Torch device. ``None`` auto-selects.

    random_state : int, optional (default=42)
        Seed forwarded to the autoencoder.

    verbose : int, optional (default=0)
        Autoencoder verbosity.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_clips,)
        Clip-level outlier scores of the training data.

    threshold_ : float
        Score threshold based on ``contamination``.

    labels_ : numpy array of shape (n_clips,)
        Binary labels of training clips (0: inlier, 1: outlier).

    ae_ : AutoEncoder
        The fitted frame-level autoencoder.

    Examples
    --------
    >>> import numpy as np
    >>> from pyod.models.audio_ae import AudioAE
    >>> clips = [np.random.RandomState(s).randn(22050) for s in range(20)]
    >>> clf = AudioAE(epoch_num=5)
    >>> clf.fit(clips)  # doctest: +SKIP
    >>> scores = clf.decision_function(clips)  # doctest: +SKIP
    """

    def __init__(self, n_mels=64, context=5, hop_length=512, sr=_DEFAULT_SR,
                 contamination=0.1, epoch_num=40, batch_size=1024, lr=1e-3,
                 hidden_neuron_list=None, device=None, random_state=42,
                 verbose=0):
        super(AudioAE, self).__init__(contamination=contamination)
        self.n_mels = n_mels
        self.context = context
        self.hop_length = hop_length
        self.sr = sr
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_neuron_list = hidden_neuron_list
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _extract(self, X):
        """Return (frames, clip_idx) over all clips in X."""
        try:
            import librosa  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            raise ImportError(
                "AudioAE requires 'librosa' and 'soundfile'. "
                "Install with: pip install pyod[audio]")
        if len(X) == 0:
            raise ValueError("AudioAE received an empty input.")
        frames_list, clip_idx = [], []
        for i, item in enumerate(X):
            y = _to_mono_waveform(item, self.sr)
            windows = _logmel_windows(y, self.sr, self.n_mels,
                                      self.context, self.hop_length)
            frames_list.append(windows)
            clip_idx.append(np.full(len(windows), i, dtype=np.int64))
        return np.concatenate(frames_list, axis=0), np.concatenate(clip_idx)

    @staticmethod
    def _aggregate(frame_scores, clip_idx, n_clips):
        """Mean per-frame score within each clip."""
        out = np.zeros(n_clips, dtype=np.float64)
        for i in range(n_clips):
            mask = clip_idx == i
            if mask.any():
                out[i] = float(frame_scores[mask].mean())
        return out

    def fit(self, X, y=None):
        """Fit the frame autoencoder and score the training clips.

        Parameters
        ----------
        X : list
            Audio clips as file paths, waveform arrays, or
            ``(waveform, sample_rate)`` tuples.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "AudioAE requires torch (for the autoencoder) and "
                "pyod[audio] (librosa, soundfile). Install with: "
                "pip install pyod[torch,audio]")
        from .auto_encoder import AutoEncoder

        frames, clip_idx = self._extract(X)
        dim = frames.shape[1]
        hidden = self.hidden_neuron_list or [128, 32, 8]
        # Drop hidden layers that are not smaller than the input so the
        # autoencoder stays a contraction for unusually small windows.
        hidden = [h for h in hidden if h < dim] or [max(dim // 2, 2)]

        # Cap the batch size to the frame count. PyOD's AutoEncoder drops
        # the last incomplete batch, so a batch larger than the dataset
        # would drop every frame and leave the training loop with nothing.
        batch_size = max(1, min(self.batch_size, frames.shape[0]))

        self.ae_ = AutoEncoder(
            contamination=self.contamination, epoch_num=self.epoch_num,
            batch_size=batch_size, lr=self.lr,
            hidden_neuron_list=hidden, device=self.device,
            random_state=self.random_state, verbose=self.verbose)
        self.ae_.fit(frames)

        frame_scores = self.ae_.decision_function(frames)
        self._set_n_classes(y)
        self.decision_scores_ = self._aggregate(frame_scores, clip_idx, len(X))
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict clip-level anomaly scores for X.

        Parameters
        ----------
        X : list
            Audio clips in the same formats accepted by ``fit``.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_clips,)
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        frames, clip_idx = self._extract(X)
        frame_scores = self.ae_.decision_function(frames)
        return self._aggregate(frame_scores, clip_idx, len(X))
