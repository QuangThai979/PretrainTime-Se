"""Detector construction from ADEngine plans.

Extracted from `pyod.utils.ad_engine.ADEngine` in 2026-05.
Not part of the public API.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import annotations

import importlib
import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyod.utils.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


def _accepts_random_state(cls: type) -> bool:
    """Return True if `cls.__init__` declares an explicit `random_state` parameter.

    A class is considered to accept ``random_state`` only when the parameter
    is named explicitly in the signature, not when it would be absorbed via
    ``**kwargs``. The conservative check prevents accidental forwarding to
    classes whose ``**kwargs`` is not safe (see pyod issue #685: ABOD / KNN /
    LUNAR / SOD forward ``**kwargs`` to sklearn NearestNeighbors which rejects
    ``random_state``).
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    for name, param in sig.parameters.items():
        if name == 'random_state' and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def build_detector_from_plan(plan: dict, kb: 'KnowledgeBase',
                             random_state: int | None = None) -> object:
    """Build and return an unfitted detector from a plan.

    Parameters
    ----------
    plan : dict (DetectionPlan)
        Output of plan_detection().
    kb : KnowledgeBase
        Knowledge base used to look up algorithm metadata.
    random_state : int or None, optional
        Random seed forwarded to the detector when the detector class
        declares an explicit ``random_state`` parameter. Detectors that do
        not declare it (e.g., ABOD, KNN, LOF, SOD) are instantiated
        without it, preserving the v3.5.1 behavior for those classes.
        If ``plan['params']`` already specifies ``random_state``, the
        plan's value wins (explicit caller intent overrides engine default).

    Returns
    -------
    detector : BaseDetector
    """
    name = plan['detector_name']
    algo = kb.get_algorithm(name)
    if algo is None:
        raise ValueError("Unknown detector '%s'" % name)
    if algo.get('status') not in ('shipped', 'experimental'):
        raise ValueError(
            "Detector '%s' has status '%s' and cannot be built"
            % (name, algo.get('status', 'unknown')))

    preset = plan.get('preset')
    if preset:
        return build_from_preset(name, preset,
                                 plan.get('params', {}),
                                 random_state=random_state)

    class_path = algo['class_path']
    module_path, class_name = class_path.rsplit('.', 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    params = dict(plan.get('params', {}))
    if (random_state is not None
            and 'random_state' not in params
            and _accepts_random_state(cls)):
        params['random_state'] = random_state
    return cls(**params)


def build_from_preset(detector_name: str, preset: str,
                      extra_params: dict,
                      random_state: int | None = None) -> object:
    """Build a detector using a factory preset.

    Presets are class-method factories that wire common defaults for
    a modality (e.g., text or image). Currently only `EmbeddingOD`
    exposes presets (``'for_text'``, ``'for_image'``).

    Parameters
    ----------
    detector_name : str
        Class name of the detector. Currently only ``'EmbeddingOD'``
        is recognized.
    preset : str
        Preset name. For ``'EmbeddingOD'``, one of ``'for_text'`` or
        ``'for_image'``.
    extra_params : dict
        Additional kwargs forwarded to the preset class method.
    random_state : int or None, optional
        Random seed forwarded to the preset factory's ``random_state``
        kwarg if the caller did not already supply one in
        ``extra_params``. EmbeddingOD presets forward this further to
        the inner detector (e.g., LUNAR), keeping
        ``ADEngine(random_state=...).build_detector(preset_plan)``
        deterministic end-to-end. Plan-level ``random_state`` in
        ``extra_params`` wins over the engine default.

    Returns
    -------
    BaseDetector
        Unfitted detector instance.

    Raises
    ------
    ValueError
        If the (detector_name, preset) pair is not recognized.
    """
    if detector_name == 'EmbeddingOD':
        from pyod.models.embedding import EmbeddingOD
        params = dict(extra_params)
        if random_state is not None and 'random_state' not in params:
            params['random_state'] = random_state
        if preset == 'for_text':
            return EmbeddingOD.for_text(**params)
        elif preset == 'for_image':
            return EmbeddingOD.for_image(**params)
        elif preset == 'for_audio':
            return EmbeddingOD.for_audio(**params)
    raise ValueError("Unknown preset '%s' for '%s'"
                     % (preset, detector_name))
