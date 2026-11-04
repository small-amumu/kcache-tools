# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The kcache-tools Authors
"""permute on a synthetic KVPM file; check argument refusals (no llama.cpp needed)."""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from kcache_tools.fix.permute import gguf_perm

ROOT = Path(__file__).resolve().parents[1]  # run from the repository root, installed or not


def run(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "kcache_tools", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_gguf_perm() -> None:
    assert gguf_perm(8).tolist() == [0, 4, 1, 5, 2, 6, 3, 7]


def test_permute_file(tmp_path: Path) -> None:
    m = np.arange(2 * 3 * 8, dtype="<f4").reshape(2, 3, 8)
    src, dst = tmp_path / "hf.kvpm", tmp_path / "gguf.kvpm"
    src.write_bytes(b"KVPM" + struct.pack("<4i", 1, 2, 3, 8) + m.tobytes())
    assert run("permute", str(src), str(dst)).returncode == 0
    raw = dst.read_bytes()
    assert raw[:20] == src.read_bytes()[:20]
    out = np.frombuffer(raw[20:], dtype="<f4").reshape(2, 3, 8)
    assert np.array_equal(out, m[..., gguf_perm(8)])


def test_check_refusals(tmp_path: Path) -> None:
    assert "needs the postmean file" in run("check", "bin", "m.gguf", "t.txt").stderr
    assert "not found" in run("check", "bin", "m.gguf", "t.txt", "--m", "missing.kvpm").stderr
    bad = run("check", "bin", "m.gguf", "t.txt", "--no-postmean", "--formats", "q2_k")
    assert "--formats must be a subset" in bad.stderr
    both = run("check", "bin", "m.gguf", "t.txt", "--m", "m.kvpm", "--no-postmean")
    assert "not both" in both.stderr


def test_help() -> None:
    out = run("--help")
    listed = {ln.split()[0] for ln in out.stdout.splitlines() if ln.startswith("  ")}
    assert out.returncode == 0 and listed == {"check", "calib", "calib-vl", "permute", "layout"}
