# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The kcache-tools Authors
"""Check that the engine's K channel order is the HF order permuted by gguf_perm, from the GGUF
weights themselves.

Reads blk.<l>.attn_k.weight from the GGUF (dequantised with gguf-py) and
model.layers.<l>.self_attn.k_proj.weight from the HF checkpoint, and compares, head by head, the
GGUF rows with the HF rows in the identity order and in the gguf_perm order (relative Frobenius
error). The order whose error is less than half the other one is reported ("gguf" for Llama-family
GGUFs, "hf" for Qwen2 / Qwen3); the exit code is 1 only when neither order fits.
Usage: python -m kcache_tools layout <gguf> <hf_model_dir> [--gguf-py DIR] [layers ...]
Needs the gguf package (pip install gguf, or pip install -e .[layout]), or llama.cpp's gguf-py
directory given with --gguf-py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

from kcache_tools.fix.permute import gguf_perm


def hf_weight(model_dir: Path, name: str) -> np.ndarray:
    index = model_dir / "model.safetensors.index.json"
    shard = (
        json.loads(index.read_text(encoding="utf-8"))["weight_map"][name]
        if index.exists()
        else "model.safetensors"
    )
    with safe_open(str(model_dir / shard), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def main() -> None:
    args = sys.argv[1:]
    if "--gguf-py" in args:
        i = args.index("--gguf-py")
        sys.path.insert(0, args[i + 1])
        del args[i : i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    try:
        from gguf import GGUFReader
        from gguf.quants import dequantize
    except ImportError:
        sys.exit(
            "kcache-tools layout needs the gguf package: pip install gguf, "
            "or pass --gguf-py <llama.cpp/gguf-py>"
        )
    gg, hf = args[0], Path(args[1])
    cfg = json.loads((hf / "config.json").read_text(encoding="utf-8"))
    nl = cfg["num_hidden_layers"]
    layers = [int(x) for x in args[2:]] or [0, nl // 2 - 1, nl - 1]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    r = GGUFReader(gg)
    tensors = {t.name: t for t in r.tensors}
    p = gguf_perm(hd)
    rows = []
    for li in layers:
        t = tensors[f"blk.{li}.attn_k.weight"]
        g = dequantize(t.data, t.tensor_type).reshape(n_kv * hd, -1).astype(np.float64)
        w = hf_weight(hf, f"model.layers.{li}.self_attn.k_proj.weight").astype(np.float64)
        assert g.shape == w.shape, (g.shape, w.shape)
        for h in range(n_kv):
            gh, wh = g[h * hd : (h + 1) * hd], w[h * hd : (h + 1) * hd]
            e_id = np.linalg.norm(gh - wh) / np.linalg.norm(wh)
            e_p = np.linalg.norm(gh - wh[p]) / np.linalg.norm(wh)
            rows.append(
                {
                    "layer": li,
                    "head": h,
                    "err_identity": round(float(e_id), 5),
                    "err_perm": round(float(e_p), 5),
                }
            )
    if all(x["err_perm"] < 0.5 * x["err_identity"] for x in rows):
        order = "gguf"
    elif all(x["err_identity"] < 0.5 * x["err_perm"] for x in rows):
        order = "hf"
    else:
        order = "unknown"
    ok = order != "unknown"
    print(
        json.dumps(
            {
                "head_dim": hd,
                "n_kv": n_kv,
                "layers": layers,
                "order": order,
                "perm_matches": order == "gguf",
                "max_err_perm": max(x["err_perm"] for x in rows),
                "min_err_identity": min(x["err_identity"] for x in rows),
            }
        )
    )
    if not ok:
        print(json.dumps(rows))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
