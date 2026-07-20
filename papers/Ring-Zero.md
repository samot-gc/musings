---
tags:
    - reasoning
    - RL
    - RLVR
    - scaling
    - training
method: Ring-Zero
title: 'Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning'
lab:
    - Ant Group
    - Gaoling School of AI
date: 202607
---

[*Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning*](https://arxiv.org/abs/2607.12395) is mainly a scale study and training recipe for zero-RL reasoning:

- Scales RL directly from a pretrained base model without reasoning SFT, though the full pipeline later includes self-distillation/SFT
- Headline model is a 1T-parameter MoE with 63B activated parameters; comparison model is 104B total / 7.4B activated
- Trains on maths problems using answer correctness plus strict `<think>...</think><answer>...</answer>` termination rewards

Their goal is not just correctness, but high-quality CoT along three dimensions:

- **Comprehensible:** coherent and easy for humans to follow
- **Reproducible:** useful as distillation data for weaker models
- **Efficient:** concise, without redundant or circular reasoning

The key aspect is changing the optimisation regime over training, rather than trying to optimise discovery and efficiency with one objective:

1. **Elicit reasoning:**
    - use token-level loss, while progressively increasing context from 4k to 64k
    - this amplifies rare, correct reasoning tokens, giving the model room to develop longer derivations
2. **Compress and reset:**
    - sample multiple traces, select and further prune the shortest correct one then distil back into the *original* base model
    - this retains the acquired reasoning whilst removing verbosity and resetting accumulated train/rollout mismatch
3. **Continue RL:**
    - switch to sample-level loss normalisation, removing bias towards longer outputs
    - this allows performance to improve without continued length inflation
4. **Control the budget:**
    - train explicit short, medium and long reasoning modes
    - this gives controllable inference cost, but slightly reduces peak long-budget performance

This resolves the apparent contradiction with their comprehensibility and efficiency goals:

1. Use length pressure to elicit reasoning
2. Compress the traces and remove the length incentive

Authors acknowledge that this is a staged heuristic, rather than a unified objective for reasoning quality and token efficiency.

Results are broadly positive, but narrow:

- 1T model improves faster per training step and reaches a higher ceiling than the 104B model, particularly on harder maths
- Pass@1024 improves early and then plateaus, while pass@1 continues rising:
  - an initial *discovery* phase unlocking latent solution paths
  - followed by *sharpening* their probability
- Self-distillation is not just compression: it improves the model and produces traces that transfer well to smaller students
- Evidence is entirely from mathematical reasoning; some reasoning-quality evaluation also relies on LLM judges

The most useful parts for post-training (according to ChatGPT):

- Token-level loss looks useful for bootstrapping, but harmful as a permanent objective: it creates severe "length inertia", including on already-solved questions
- Train/rollout numerical mismatch is a first-order collapse mode
  - compute the importance-ratio numerator from training-engine logits
  - run attention softmax and the LM head in FP32 while leaving the rest in BF16
- Loose output formatting gets reward-hacked into endless trailing text; correct termination needs to be part of the reward
- Training data should become harder as the model improves, rather than preserving the natural long tail of easy examples

Overall, a useful engineering paper, though the framing is stronger than the evidence:

- Scale result compares two models, not a scaling law; detailed ablations use the smaller model and "sample efficiency" is not compute-normalised
- Claim that hand-crafted heuristics become redundant is overstated: the pipeline still uses format rewards, context curricula, trace filtering and explicit budget prompts
- The interesting "emergent" behaviours are better described as structured decomposition, self-checking and branching; "context anxiety" is essentially budget-aware reward hacking
