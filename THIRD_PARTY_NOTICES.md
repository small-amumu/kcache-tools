# Third-party notices (tools)

This project's own code is licensed under Apache-2.0 (see LICENSE and NOTICE). The following parts follow
third-party work:

- **llama.cpp** (ggml-org/llama.cpp, MIT License, Copyright (c) 2023-2026 The ggml authors):
  `kcache_tools/fix/permute.py` re-implements the q/k channel permutation of the GGUF converter; no llama.cpp source
  is copied into it. `patches/llama.cpp-3bb63cb-postmean.patch` applies to llama.cpp commit
  3bb63cbcbfb62056c84fe9e521b0f3d756d29c32 and contains unchanged context lines from it; the full MIT License
  text is in `patches/LICENSE-llama.cpp.txt`. The changes in the patch are Apache-2.0.
