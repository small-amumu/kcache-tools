# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The kcache-tools Authors
"""Command-line entry point: python -m kcache_tools <command> [args].

Commands (each prints its own usage with -h, or with no arguments for the positional-argument
tools):
  check      pre-deployment check on llama.cpp
  calib      static post-RoPE key mean -> KVPM file (postmean repair)
  calib-vl   the same for the Qwen2.5-VL text decoder
  permute    reorder a KVPM file to the GGUF order of Llama models
  layout     verify the engine's K channel order against the weights
"""

from __future__ import annotations

import runpy
import sys

COMMANDS = {
    "check": "kcache_tools.check.check",
    "calib": "kcache_tools.fix.calib",
    "calib-vl": "kcache_tools.fix.calib_vl",
    "permute": "kcache_tools.fix.permute",
    "layout": "kcache_tools.fix.layout",
}


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    sys.argv = [f"kcache-tools {cmd}"] + sys.argv[2:]
    runpy.run_module(COMMANDS[cmd], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
