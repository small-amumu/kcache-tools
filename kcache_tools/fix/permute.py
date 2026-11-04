# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The kcache-tools Authors
"""Reorder a KVPM file (`kcache-tools calib` output, HF channel order) into the GGUF channel order of
Llama-family models.

The GGUF converter permutes the Llama q / k projections per head, so engine channel 2j + s holds
HF channel s * hd / 2 + j; the file is reordered by p[2j + s] = s * hd / 2 + j. Qwen2 / Qwen3 are
not permuted by the conversion and must not be passed through this script.
Usage: python -m kcache_tools permute <in_hf.kvpm> <out_gguf.kvpm>
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np


def gguf_perm(head_dim: int) -> np.ndarray:
    half = head_dim // 2
    return np.array([s * half + j for j in range(half) for s in range(2)], dtype=np.int64)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    raw = src.read_bytes()
    assert raw[:4] == b"KVPM", "not a KVPM file"
    ver, n_layer, n_head_kv, head_dim = struct.unpack("<4i", raw[4:20])
    assert head_dim % 2 == 0, f"head dim {head_dim} is odd"
    assert (len(raw) - 20) % 4 == 0, "KVPM payload is not float32"
    assert ver == 1, f"KVPM version {ver}"
    data = np.frombuffer(raw[20:], dtype="<f4")
    assert data.size == n_layer * n_head_kv * head_dim, "KVPM size does not match its header"
    m = data.reshape(n_layer, n_head_kv, head_dim)
    out = np.ascontiguousarray(m[..., gguf_perm(head_dim)])
    with dst.open("wb") as f:
        f.write(raw[:20])
        f.write(out.astype("<f4").tobytes())
    print(
        f"{src} -> {dst}: L={n_layer} H_kv={n_head_kv} D={head_dim}, "
        "channels reordered to the GGUF order"
    )


if __name__ == "__main__":
    main()
