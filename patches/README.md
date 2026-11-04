# llama.cpp patch: static post-RoPE key mean (postmean)

Adds `LLAMA_KV_POSTMEAN=<file>` to llama.cpp: the model graph subtracts a calibrated static per-head key mean
(a KVPM file from `kcache-tools calib` or `kcache-tools calib-vl`) from K after RoPE, before the K-cache write. The patch
changes only the model graphs and has no backend-specific code.

## Base
llama.cpp commit `3bb63cbcbfb62056c84fe9e521b0f3d756d29c32` (2026-10-20).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 3bb63cbcbfb62056c84fe9e521b0f3d756d29c32
git apply /path/to/llama.cpp-3bb63cb-postmean.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # add -DGGML_CUDA=ON for CUDA
cmake --build build -j --target llama-perplexity llama-server llama-cli
```

## What the patch changes

| File | Change |
|---|---|
| `src/models/kv-pm.h` (new) | KVPM reader and the graph input holding the mean; `kv_pm::sub` subtracts the mean of layer l from K |
| `src/models/qwen2.cpp`, `src/models/qwen3.cpp`, `src/models/llama.cpp`, `src/models/qwen2vl.cpp` | Call `kv_pm::sub` on K after RoPE when `LLAMA_KV_POSTMEAN` is set |

## Runtime switch

| Variable | Values | Effect |
|---|---|---|
| `LLAMA_KV_POSTMEAN` | file | Static post-RoPE key mean (`KVPM` format, see `kv-pm.h`); hooked into the qwen2, qwen3, llama and qwen2vl graphs (ignored by other architectures). Supported for graphs without attention sinks or logit softcapping; do not combine it with context shifting (`--context-shift`). A file whose shape does not match the model aborts. Unset: the engine is unchanged. |

## License
Our changes (the added and modified lines of the patch, including the new file `kv-pm.h`) are licensed under
the Apache License 2.0, like the rest of this repository (see `LICENSE` and `NOTICE` at the repository root).
The patch applies to llama.cpp and contains unchanged context lines from it; llama.cpp is distributed under the
MIT License, Copyright (c) 2023-2026 The ggml authors. Its license text is reproduced in `LICENSE-llama.cpp.txt`
in this directory and must be kept with any copy of the patched files.
