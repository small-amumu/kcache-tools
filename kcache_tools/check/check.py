# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The kcache-tools Authors
"""Pre-deployment check for llama.cpp's quantised K cache.

For each candidate K-cache format (smallest first) the check measures D, the mean output KL
divergence against the 16-bit K cache over the first 8192 tokens of the given text, in four
2048-token chunks (the second half of each chunk is scored; llama-perplexity --kl-divergence,
-c 2048, 4 chunks, V q8_0); a text shorter than 8192 tokens is refused. D < 0.1016 allows the
format; otherwise postmean (LLAMA_KV_POSTMEAN=<KVPM file from `kcache-tools calib` / `calib-vl`>) is
tried and allowed under the same threshold; a repaired D' > 0.3577 raises an alarm. If nothing
passes, q8_0 is returned when it passes, else the 16-bit cache. The result is the "decision" field
of the JSON on stdout (the exit status is 0 whenever a decision is made). The decision is measured
on text only, at a 2048-token context (second half of each chunk scored), with flash attention on
and a q8_0 V cache ("measured_with" in the JSON); longer contexts, image inputs and other settings
are not covered.
Usage:
  python -m kcache_tools check <llama.cpp bin dir> <model.gguf> <text> (--m m.kvpm | --no-postmean)
      [--formats q4_0,q4_1] [--ngl 99]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TAU_ALLOW = 0.1016  # allow threshold on D
TAU_ALARM = 0.3577  # alarm threshold on the repaired D'
PATCH_ENV = ("LLAMA_KV_POSTMEAN",)  # never inherited from the caller's shell
BPW = {"q4_0": 4.5, "q4_1": 5.0, "q8_0": 8.5}  # bits per weight; the formats the check supports
REPORT_ENV = ("LLAMA_ATTN_ROT_DISABLE",)  # recorded in the output when set
KLD = re.compile(r"^Mean\s+KLD:\s+([0-9.eE+-]+)", re.M)
CHUNKS = re.compile(r"over (\d+) chunks")
SAME_D = 1e-6  # a postmean D this close to the plain D means the patch did not run
MEASURED_WITH = {"input": "text", "flash_attn": "on", "ctv": "q8_0", "ctx": 2048, "chunks": 4}


def perplexity(
    args: argparse.Namespace, base: Path, ctk: str, score: bool, m: str | None = None
) -> float | None:
    cmd = [
        str(Path(args.bin) / "llama-perplexity"),
        "-m",
        args.gguf,
        "-f",
        args.text,
        "-c",
        "2048",
        "--chunks",
        "4",
        "-fa",
        "on",
        "-ctv",
        "q8_0",
        "-ngl",
        str(args.ngl),
        "-ctk",
        ctk,
        "--kl-divergence-base",
        str(base),
    ] + (["--kl-divergence"] if score else [])
    env = {k: v for k, v in os.environ.items() if k not in PATCH_ENV}
    if m:
        env["LLAMA_KV_POSTMEAN"] = m
    out = subprocess.run(
        cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    log = out.stdout + "\n" + out.stderr
    if out.returncode != 0:
        sys.exit(f"llama-perplexity failed for {ctk}: {log[-500:]}")
    chunks = CHUNKS.search(log)
    if chunks is None or int(chunks.group(1)) != 4:
        sys.exit(f"the text must give 4 chunks of 2048 tokens (at least 8192 tokens): {log[-500:]}")
    if not score:
        return None
    hit = KLD.search(log)
    if hit is None:
        sys.exit(f"no 'Mean KLD' line for {ctk}: {log[-500:]}")
    return float(hit.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(prog="kcache-tools check")
    ap.add_argument("bin")
    ap.add_argument("gguf")
    ap.add_argument("text")
    ap.add_argument(
        "--m",
        help="postmean file (KVPM) from `kcache-tools calib` / `calib-vl`; "
        "without it postmean is not tried",
    )
    ap.add_argument("--no-postmean", action="store_true", help="explicitly run without postmean")
    ap.add_argument("--formats", default="q4_0,q4_1")
    ap.add_argument("--ngl", type=int, default=99)
    args = ap.parse_args()
    if args.m and args.no_postmean:
        sys.exit("pass either --m <file> or --no-postmean, not both")
    if not args.m and not args.no_postmean:
        sys.exit(
            "the check needs the postmean file: pass --m <file> "
            "(or --no-postmean to skip the repair step)"
        )
    if args.m and not Path(args.m).exists():
        sys.exit(f"postmean file not found: {args.m}")
    trace: list[dict[str, object]] = []
    alarms: list[str] = []
    plain: dict[str, float | None] = {}

    def report(decision: str, postmean: bool) -> None:
        ran = any(t.get("postmean") and "refused" not in t for t in trace)
        print(
            json.dumps(
                {
                    "decision": decision,
                    "postmean": postmean,
                    "alarms": alarms,
                    "postmean_ran": ran,
                    "measured_with": {**MEASURED_WITH, "ngl": args.ngl},
                    "env": {k: os.environ[k] for k in REPORT_ENV if k in os.environ},
                    "trace": trace,
                }
            )
        )

    formats = args.formats.split(",")
    if any(f not in BPW for f in formats):
        sys.exit(f"--formats must be a subset of {sorted(BPW)}")
    formats.sort(key=BPW.get)  # smallest first
    exe = Path(args.bin) / "llama-perplexity"
    if not exe.exists() and not exe.with_suffix(".exe").exists():
        sys.exit(f"llama-perplexity not found in {args.bin}")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base.kld"
        perplexity(args, base, "f16", score=False)
        for f in formats:
            d = plain[f] = perplexity(args, base, f, score=True)
            trace.append({"format": f, "D": d})
            if d is not None and d < TAU_ALLOW:
                report(f, False)
                return
            if args.m:
                d = perplexity(args, base, f, score=True, m=args.m)
                trace.append({"format": f, "postmean": True, "D": d})
                if d is not None and abs(d - trace[-2]["D"]) <= SAME_D:
                    trace[-1]["refused"] = "patch did not run (D unchanged)"
                    print(
                        f"{f}: postmean had no effect; is llama.cpp built with the patch, and is "
                        "the architecture one of qwen2, qwen3, llama, qwen2vl?",
                        file=sys.stderr,
                    )
                    continue
                if d is not None and d < TAU_ALLOW:
                    report(f, True)
                    return
                if d is not None and d > TAU_ALARM:  # alarm only on the repaired D'; then fall back
                    trace[-1]["alarm"] = True
                    alarms.append(f)
                    print(
                        f"ALARM: {f} with postmean has D' = {d:.4f} > {TAU_ALARM}", file=sys.stderr
                    )
        if "q8_0" in plain:
            d8 = plain["q8_0"]
        else:
            d8 = perplexity(args, base, "q8_0", score=True)
            trace.append({"format": "q8_0", "D": d8})
        decision = "q8_0" if d8 is not None and d8 < TAU_ALLOW else "f16"
        report(decision, False)


if __name__ == "__main__":
    main()
