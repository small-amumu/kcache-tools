# kcache-tools

Tools for deploying llama.cpp with a low-bit K cache (`q4_0`, `q4_1`).

- **`check`** tells you, before deployment, which K-cache format a given model can safely use.
- **postmean** is a small llama.cpp patch plus a calibration tool. It lets models that fail the check run with a 4-bit K cache.

## Requirements

- Python ≥ 3.10, PyTorch, transformers
- llama.cpp at commit `3bb63cb`, built with the patch in `patches/` (needed for postmean). `check --no-postmean` also works on an unpatched build.

## Install

```bash
pip install -e .            # add [layout] for the GGUF channel-order check
```

## Build llama.cpp with the patch

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 3bb63cbcbfb62056c84fe9e521b0f3d756d29c32
git apply /path/to/kcache-tools/patches/llama.cpp-3bb63cb-postmean.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # add -DGGML_CUDA=ON for CUDA
cmake --build build -j --target llama-perplexity llama-server llama-cli
```

## Quick start

1. **Calibrate** the key mean from the Hugging Face checkpoint, using any text of at least 2048 tokens:

   ```bash
   kcache-tools calib <hf_model_dir> <calib_text> m.kvpm
   ```

   Llama and Mistral GGUFs use a different channel order, so reorder the file first:

   ```bash
   kcache-tools permute m.kvpm m_gguf.kvpm
   ```

2. **Check** the GGUF model, using a text of at least 8192 tokens:

   ```bash
   kcache-tools check <llama.cpp/build/bin> <model.gguf> <text> --m m.kvpm
   ```

   It prints a JSON result. `decision` is the K-cache format to use; `postmean` says whether the mean file is required.

3. **Deploy** with that format, flash attention on and a `q8_0` V cache. If `postmean` is `true`, also set `LLAMA_KV_POSTMEAN=m.kvpm`:

   ```bash
   LLAMA_KV_POSTMEAN=m.kvpm ./build/bin/llama-server -m model.gguf -fa on -ctk q4_1 -ctv q8_0
   ```

## Accuracy

The tables below come from llama.cpp `3bb63cb` with the patch, q8_0 GGUF weights, a `q8_0` V cache, flash attention, a 2048-token context and 4 chunks of English long-form text. The reference run keeps the K cache in f16. Calibration for postmean used a separate text.

**Top-1 agreement with the f16-K run** (share of positions where both runs predict the same next token):

| Model | q8_0 | q4_1 | q4_1 + postmean | q4_0 | q4_0 + postmean | `check` picks |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 97.8% | 3.5% | 95.4% | 6.5% | 95.0% | q4_0 + postmean |
| Qwen2.5-1.5B-Instruct | 96.7% | 3.1% | 93.3% | 4.3% | 92.6% | q4_0 + postmean |
| Qwen3-8B | 98.2% | 92.7% | 94.4% | 91.9% | 94.7% | q4_0 |
| Qwen3-1.7B | 97.1% | 81.1% | 88.7% | 76.2% | 87.8% | q4_0 + postmean |
| Llama-3.1-8B-Instruct | 98.8% | 94.3% | 95.1% | 93.8% | 93.2% | q4_0 |
| Llama-3.2-3B-Instruct | 98.7% | 93.9% | 93.8% | 93.4% | 93.0% | q4_0 |

**Perplexity:**

| Model | f16 | q8_0 | q4_1 | q4_1 + postmean | q4_0 | q4_0 + postmean |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 13.35 | 13.34 | 7606 | 13.46 | 10185 | 13.44 |
| Qwen2.5-1.5B-Instruct | 16.94 | 17.01 | 19260 | 17.09 | 6298 | 17.07 |
| Qwen3-8B | 17.82 | 17.83 | 17.99 | 17.85 | 17.51 | 17.87 |
| Qwen3-1.7B | 28.37 | 28.41 | 31.72 | 29.41 | 42.08 | 29.87 |
| Llama-3.1-8B-Instruct | 13.15 | 13.14 | 13.27 | 13.29 | 13.22 | 13.32 |
| Llama-3.2-3B-Instruct | 17.79 | 17.79 | 18.05 | 18.18 | 18.02 | 18.04 |

Both Qwen2.5 models break with a plain 4-bit K cache, and postmean brings them back to 92–95% top-1 agreement, the same range the Llama models reach with plain q4_0. On models that already pass, `check` keeps plain q4_0 and does not turn postmean on.

## Scope

- **Architectures:** postmean applies to the qwen2, qwen3, llama and qwen2vl graphs. Other architectures ignore it, and `check` reports when the patch had no effect.
- **What the decision covers:** text input at a 2048-token context, with flash attention and a `q8_0` V cache.
- **Not supported:** using postmean with `--context-shift`, or on models with attention sinks or logit softcapping.

Every command, option and threshold is described in [`TOOLS.md`](TOOLS.md).

## License

Apache License 2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The llama.cpp patch also contains context lines from llama.cpp, which is MIT-licensed; see [`patches/README.md`](patches/README.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

A paper describing this work is in preparation. Until it is out, please cite this repository.
