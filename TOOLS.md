# kcache-tools

Two tools for llama.cpp's low-bit K cache (`q4_0`, `q4_1`, `q8_0`): a pre-deployment check that decides,
per model, which K format can be deployed, and the static-mean repair (postmean) that the check tries when a
format fails.

The repair needs llama.cpp built with the patch in `patches/` (it adds `LLAMA_KV_POSTMEAN`); `check
--no-postmean` also runs on a stock build. Postmean reads a KVPM file through `LLAMA_KV_POSTMEAN`; the patch
hooks it into the qwen2, qwen3, llama and qwen2vl graphs only, and other architectures ignore it (the check
notices this: a D that does not change under postmean is reported as "patch did not run"). Do not use
postmean together with context shifting (`--context-shift`).

Install: `pip install -e .` (Python >= 3.10, PyTorch, transformers; `pip install -e .[layout]` adds the `gguf`
package for `layout`). Every tool runs as `python -m kcache_tools <command>` (or `kcache-tools <command>` after
installation); `python -m kcache_tools --help` lists the commands. The calibration tools do not run a checkpoint's
own Python code.

## Usage

```bash
# 1. calibrate the mean on a text of at least 2048 tokens (Hugging Face checkpoint; calib-vl for
#    Qwen2.5-VL)
python -m kcache_tools calib <hf_model_dir> <calib_text> m.kvpm
M=m.kvpm
# 1b. Llama / Mistral GGUFs only: reorder to the GGUF channel order (layout tells which order a GGUF
#     uses)
python -m kcache_tools permute m.kvpm m_gguf.kvpm
M=m_gguf.kvpm
# 2. check the GGUF on a text of at least 8192 tokens with the patched llama.cpp
python -m kcache_tools check <llama.cpp/build/bin> <model.gguf> <text> --m $M
# 3. deploy the "decision" K format with flash attention on and a q8_0 V cache, with
#    LLAMA_KV_POSTMEAN=$M when "postmean" is true (the scope of the decision: see check below)
```

## Check (llama.cpp)

| Command | What it does |
|---|---|
| `check` | The pre-deployment check. For each candidate K format (`--formats`, default `q4_0,q4_1`, sorted smallest first), it measures D, the mean output KL against the 16-bit K cache (`llama-perplexity --kl-divergence`, 4 chunks of 2048 tokens, V `q8_0` in both runs). D < 0.1016 allows the format; otherwise, with `--m`, it retries with the postmean repair; a repaired D' > 0.3577 raises an alarm. If nothing passes it returns `q8_0` when its D passes, else the 16-bit cache. The result is the JSON on stdout (`decision`, `postmean`, `alarms`, `postmean_ran`, `measured_with`, and every measured D in `trace`); the exit status is 0 whenever a decision is made. The decision is measured on text only, at a 2048-token context (second half of each chunk scored), with flash attention on and a `q8_0` V cache (`measured_with`); longer contexts, image inputs and other settings are not covered by it. The output records `LLAMA_ATTN_ROT_DISABLE` when it is set. |

## Repair

| Command | What it does |
|---|---|
| `calib`, `calib-vl` | Static post-RoPE key mean per layer and KV head, written as a KVPM file for `LLAMA_KV_POSTMEAN`. The patched engine subtracts it from K after RoPE, before the cache write. Supported for graphs without attention sinks or logit softcapping, and not together with context shifting. |
| `permute`, `layout` | Llama-family GGUFs permute the q / k channels per head. `calib` writes the Hugging Face order, so for Llama / Mistral the file must be reordered with `permute` (calib prints a note); a file in the wrong order is accepted by the engine silently and only shows up as a worse D'. `layout` reports which order a GGUF's K weights use. |

## Thresholds

| Constant | Used by |
|---|---|
| D < 0.1016 (allow) | `check` |
| D' > 0.3577 (alarm), on the repaired D' only | `check` |

Models and texts are not included; the tools read them from paths you give.
