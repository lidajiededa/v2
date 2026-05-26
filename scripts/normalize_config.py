#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PANGU_V2_ARCH = "PanguV2MoEForCausalLM"


def normalize(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["architectures"] = [PANGU_V2_ARCH]
    cfg["model_type"] = "pangu_v2_moe"

    rope_theta = cfg.get("rope_theta", 6400000)
    rope_parameters = cfg.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        rope_parameters = {"rope_type": "default"}
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters["rope_theta"] = rope_theta
    cfg["rope_parameters"] = rope_parameters

    cfg.setdefault("pad_token_id", cfg.get("eos_token_id"))

    # vLLM's hybrid checks look for non-attention layer markers. The MOME
    # state is attached to attention modules, so keep one sentinel marker at
    # the end instead of inventing fake decoder layers.
    if cfg.get("use_mome") and "layer_types" not in cfg:
        cfg["layer_types"] = ["attention"] * int(cfg["num_hidden_layers"]) + ["mome"]

    # The MTP layers are present in weights/config, but the first bring-up path
    # serves the base model only. The model loader skips those layers.
    cfg.setdefault("_pangu_v2_notes", {})
    cfg["_pangu_v2_notes"].update(
        {
            "normalized_by": "scripts/normalize_config.py",
            "target_architecture": PANGU_V2_ARCH,
            "bringup_mode": "base_model_first; mtp_weights_skipped_by loader",
        }
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    out = normalize(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()

