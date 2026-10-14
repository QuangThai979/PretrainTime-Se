Audio Detectors
=================

PyOD detects audio anomalies through two paths: the lightweight ``EmbeddingOD.for_audio()`` (handcrafted acoustic features run through any detector, see :doc:`pyod.models.embedding`) and the dedicated :class:`~pyod.models.audio_ae.AudioAE` deep detector below. Install with ``pip install pyod[audio]``.

pyod.models.audio\_ae module
----------------------------------

.. automodule:: pyod.models.audio_ae
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:
