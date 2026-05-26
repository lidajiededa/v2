from __future__ import annotations

from typing import Any


PANGU_V2_ARCH = "PanguV2MoEForCausalLM"
PANGU_V2_MODEL_TYPE = "pangu_v2_moe"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def normalize_pangu_v2_config(config: Any) -> None:
    """Normalize known Pangu V2 MoE config wrinkles in-place.

    The checkpoint config used by OpenPangu 505B carries `rope_theta` at the
    top level and may have an old architecture alias. vLLM 0.18 reads
    `rope_parameters["rope_theta"]` in the OpenPangu MLA path, so normalize it
    before model construction.
    """

    model_type = _get(config, "model_type")
    archs = list(_get(config, "architectures", []) or [])
    is_pangu_v2 = model_type == PANGU_V2_MODEL_TYPE or PANGU_V2_ARCH in archs
    if not is_pangu_v2:
        return

    if not archs or archs != [PANGU_V2_ARCH]:
        _set(config, "architectures", [PANGU_V2_ARCH])

    rope_theta = _get(config, "rope_theta", 10000)
    rope_parameters = _get(config, "rope_parameters")
    if not isinstance(rope_parameters, dict):
        rope_parameters = {"rope_type": "default"}
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters["rope_theta"] = rope_theta
    _set(config, "rope_parameters", rope_parameters)

    if _get(config, "pad_token_id") is None:
        # Omni's Pangu model uses pad_token_id to construct embeddings. The
        # generation config does not define a dedicated pad token, so EOS is
        # the least surprising default.
        _set(config, "pad_token_id", _get(config, "eos_token_id"))

    # vLLM's hybrid detection expects a non-attention marker on hybrid models.
    # Keep the layer count equal to num_hidden_layers; MTP layers are handled
    # separately by the model loader.
    if _get(config, "use_mome", False) and _get(config, "layer_types") is None:
        num_layers = int(_get(config, "num_hidden_layers", 0) or 0)
        _set(config, "layer_types", ["attention"] * num_layers + ["mome"])


def is_pangu_v2_moe_config(config: Any) -> bool:
    return (
        _get(config, "model_type") == PANGU_V2_MODEL_TYPE
        or PANGU_V2_ARCH in (_get(config, "architectures", []) or [])
    )

