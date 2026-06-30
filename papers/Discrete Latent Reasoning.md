---
tags:
    - continuous thoughts
    - latent reasoning
    - reasoning
    - RL
    - SFT
method: DLR
title: 'Why Struggle with Continuous Latents? Interpretable Discrete Latent Reasoning via Rendered Compression'
lab:
    - Shanghai Jiao Tong University
    - Tongji University
date: 202606
---

[*Why Struggle with Continuous Latents?*](https://arxiv.org/abs/2606.29712) proposes *Discrete Latent Reasoning*:

- TL;DR:
  - Replace explicit CoT with a short sequence of non-verbal tokens via image tokeniser
  - Kinda insane, unprincipled idea (patches may not even cover full words!), but decent performance

- Constructs these by compressing real CoTs:
  - take written CoT, render as image ("screenshot"), tokenise with DeepEncoder V2
  - reduces 1024 × 1024 image into 256 tokens, leading to up to 20× compression

- Codebook trained so that latent tokens can be (approximately) decoded:
  - is a code from a learned compression scheme over actual CoTs
  - has an approximate textual interpretation via the decoder
  - can be supervised with ordinary token-level losses

- Training the LM is then fairly clean:
  1. Align the new latent tokens to the LM: learn projectors from codebook to LM space
  2. SFT on sequences like `problem -> <latent> compressed-CoT tokens </latent> -> answer`
  3. RL the model's latent-token policy: correctness, formatting and/or process-level via decode

[*Abstract-CoT*](https://arxiv.org/abs/2604.22709) has similar goals, with discrete latents whose embeddings are learned, but differs in the implementation:

- Abstract-CoT: learns (reserved) latent language from scratch during training
- DLR: token IDs precomputed from compressed images

Results are broadly positive:

- DLR beats latent-reasoning baselines, but still a step below standard CoT
- It uses *much* shorter traces than explicit CoT - perhaps 20x shorter
- RL helps, and ablations suggest alignment stage, SFT stage and process reward all matter
