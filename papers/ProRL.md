---
tags:
    - grpo
    - reasoning
    - rl
    - training
method: grpo
parent: 'ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models'
authors: NVIDIA
date: 202505
---

# ProRL Summary

-   [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864)
-   2025-05; NVIDIA

## High-Level Summary

-   Introduces Nemotron-Research-Reasoning-Qwen-1.5B, a reasoning model based on DeepSeek-R1-1.5B
-   Nemotron-1.5B outperforms its 1.5B base model, and even sometimes the DeepSeek-R1-7B
-   Introduces *ProRL*, a prolonged RL training framework
-   Argues that this allows the uncovering of novel reasoning strategies inaccessible to the base model
-   Observed RL gain is larger when base model performs poorly, even increasing from ~0% base to ~100% after RL


## Elevator Pitch

Recent advancements in reasoning have focused on RL-based fine-tuning. A fundamental question remains under debate:

>   Does RL truly unlock new reasoning capabilities, or does it merely optimise sampling of solutions already learned?

Several recent studies argue the latter, basing their conclusions on pass@$k$ metrics with large $k$. The authors of this paper argue the former, positing that the previous conclusions may stem from *methodological* constraints, not fundamental limitations of RL.

1.  Overreliance on specialised domains in which the models are often overtrained during both pre- and post-training, restricting potential for RL exploration.
2.  Premature termination of RL training prior to full exploration and development of new reasoning capabilities.

![ProRL overview](attachments/ProRL%20-%20Overview.png)

## Methodology

The RL training is based on the GRPO algorithm, with a few tweaks to address entropy collapse. Entropy collapse, where the model's output becomes too concentrated early, causes the policy to commit to a narrow set of outputs prematurely, limiting exploration. This is particularly bad in GRPO, where the learning signal relies on having a diverse set of sampled outputs.

1.  <details open>
    <summary><i>Decoupled Clipping.</i></summary>

    In the original GRPO, the clipping is uniform. Here, the upper and lower thresholds are separated: $\operatorname{clip}(r_\theta, 1 - \varepsilon_\textsf{low}, 1 + \varepsilon_\textsf{high})$, where $r_\theta$ is the observed policy ratio. Increasing $\varepsilon_\textsf{high}$ further prompts the model to uplift probabilities of unlikely tokens which, nevertheless, provided significant advantage. They take $\varepsilon_\textsf{low} = 0.2$ (the default for $\varepsilon$ in many libraries) but $\varepsilon_\textsf{high} = 0.4$. This is pretty high: [*Magistral*](http://arxiv.org/abs/2506.10910) take $\varepsilon_\textsf{high} \approx 0.25$, for comparison.
    </details>

2.  <details open>
    <summary>Token-Level Policy Gradient Loss.</i></summary>
    
    GRPO employs a *sample-level* loss calculation: first, average the losses by token within each sample; then, aggregate the losses across samples. Notationally, $G^{-1} \sum_{i=1}^G |o_i|^{-1} \sum_{t=1}^{|o_i|} (...)$. Each *sample* is assigned equal weight, and so *tokens* in longer responses are underweighted. DAPO uses a *token-level* calculation: $(\sum_{i=1}^G |o_i|)^{-1} \sum_{i=1}^G \sum_{t=1}^{|o_i|} (...)$.
    </details>

These are taken from [DAPO](https://arxiv.org/abs/2503.14476).

3.  <details open>
    <summary><i>KL Regularisation and Reference-Policy Reset.</i></summary>

    The original GRPO algorithm *does* include a KL penalty term $- \beta \operatorname{KL}(\pi_\theta \mathrel{\|} \pi_\textsf{ref})$, where $\pi_\textsf{ref}$ is a *fixed* reference policy—typically the pre-RL model. Several recent papers, including [*Magistral*](http://arxiv.org/abs/2506.10910) and [DAPO](https://arxiv.org/abs/2503.14476), removed this, arguing that the models naturally diverge from the reference policy anyway. As such, as training increases, the KL penalty may dominate the loss.
    
    ProRL keeps the penalty term. To alleviate domination, the reference policy $\pi_\textsf{ref}$ is hard-reset to a more recent snapshot periodically. (The hope is that) this allows continued improvement whilst maintaining the benefits of KL regularisation.
    </details>

4.  <details open>
    <summary><i>Prolonged RL.</i></summary>
    Typically RL implementations run for no more than a few hundred steps. Contrastingly, ProRL demonstrated continued improvement beyond 2000 training steps; see Figure 1 (left) above. This gave the algorithm sufficient time to explore and uncover new strategies—so they hypothesise.


### Results: Nemotron-Research-Reasoning-Qwen-1.5B

The result is, in their words, "the world's best 1.5B reasoning model"—and that is perhaps justified. Their base model is DeepSeek-R1-Distill-Qwen-1.5B.

![Nemotron evaluation](attachments/ProRL%20-%20Tables.png)

---

incomplete... to be continued...