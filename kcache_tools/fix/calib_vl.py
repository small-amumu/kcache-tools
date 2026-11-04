# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The kcache-tools Authors
"""Static post-RoPE key mean for Qwen2.5-VL (text decoder), for LLAMA_KV_POSTMEAN on the qwen2vl
graph.

Same definition and file layout as `kcache-tools calib`: m[l][h] = mean over calibration positions 0..T-1
of the post-RoPE key of layer l, kv head h (after the k bias), from a text-only forward pass.
The engine subtracts the same m from the keys of image tokens.
Usage: python -m kcache_tools calib-vl <hf_model_dir> <calib_text> <out.kvpm> [--T 2048]
       [--device cpu|cuda]
"""

import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def arg(name: str, default: int) -> int:
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


@torch.no_grad()
def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    model_dir, text_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    n_tok = arg("--T", 2048)
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
        .to(dev)
        .eval()
    )
    layers = model.model.language_model.layers
    n_layer = len(layers)
    mod = sys.modules[type(layers[0].self_attn).__module__]
    orig = mod.apply_multimodal_rotary_pos_emb
    m: dict[int, torch.Tensor] = {}
    calls = {"i": 0}

    def wrapped(
        q: torch.Tensor, k: torch.Tensor, *a: Any, **kw: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = calls["i"] % n_layer
        calls["i"] += 1
        qo, ko = orig(q, k, *a, **kw)
        m[layer] = ko.float().mean(dim=2)[0]  # [B, H_kv, T, D] -> [H_kv, D]
        return qo, ko

    mod.apply_multimodal_rotary_pos_emb = wrapped
    ids = (
        tok(text_path.read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, :n_tok].to(dev)
    )
    assert ids.shape[1] == n_tok, f"calibration text too short: {ids.shape[1]} < {n_tok} tokens"
    t0 = time.time()
    model(input_ids=ids)
    assert calls["i"] == n_layer and len(m) == n_layer, (
        f"hook saw {calls['i']} calls, expected {n_layer}"
    )
    n_head_kv, head_dim = m[0].shape
    data = torch.stack([m[i] for i in range(n_layer)]).cpu().contiguous()  # [L, H_kv, D]
    with out_path.open("wb") as f:
        f.write(b"KVPM")
        f.write(struct.pack("<4i", 1, n_layer, n_head_kv, head_dim))
        f.write(data.numpy().astype("<f4").tobytes())
    print(
        json.dumps(
            {
                "model": model_dir,
                "T": n_tok,
                "n_layer": n_layer,
                "n_head_kv": int(n_head_kv),
                "head_dim": int(head_dim),
                "m_norm_max": round(float(data.norm(dim=-1).max()), 3),
                "sec": round(time.time() - t0, 1),
                "out": str(out_path),
            }
        )
    )


if __name__ == "__main__":
    main()
