---
tags:
    - efficiency
    - inference
    - latent reasoning
    - reasoning
    - RL
    - test-time
    - training-free
method: LTPO
title: 'Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization'
lab:
    - University of Oxford
    - UCL
    - Chinese Academy of Sciences
date: 202510
---

# Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization

-   [Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](https://arxiv.org/abs/2510.04182)
-   2025-10; Wengao Ye, Yan Liang, Shan

[TOC]


## High-Level Summary

-   *Latent Thought Policy Optimization* (LTPO):
    -   Initialise latent thought $z$ to append to prompt $x$
    -   Use RL based on intrinsic confidence reward to adjust $z$
    -   Parameter-free - LLM is frozen, RL only touches $z$ not the LLM
-   Performance of LTPO:
    -   Matches or exceeds multiple baselines
    -   Significant speed-up vs CoT on complex problems (eg, ~50% on AIME)


## Elevator Pitch

Chain of thought (CoT) was pretty revolutionary, but has many issues - perhaps foremost, its cost/latency. Recently, focus has shifted from *text-based* to *latent* reasoning. However, these often struggle on challenging, OOD tasks - those in which robust reasoning is most valuable.

Enter *Latent Thought Policy Optimization* (LTPO):

-   a (fixed-size) 'latent thought' $z$ is initialised randomly;
-   RL, with a confidence-based reward, is used to optimise this $z$.

No training of the LLM is needed. In fact, AR decoding isn't even needed for the reward. This makes each RL step rapid.

Performance-wise, LTPO frequently matches full COT for the models studied (typically order 10B) for accuracy, yet is much faster on more challenging questions.


## Method

![Overview of LTPO](attachments/LTPO%20-%20Figure%201.png)

Let $\mathcal M$ denote the frozen LLM, and $E$ its embedding layer. To enable latent reasoning, the embedded prompt is concatenated with $K$ placeholder *latent thought tokens*, denoted $H$ and initialised as
\(
    H^0
=   E([\text{THINK}], ..., [\text{THINK}]).
\)
It is $H$ that is optimised at test time by RL.

-   *State* - the state is that of the latent thought tokens $H$.
-   *Action* - an action $A$ is a candidate for the next state of latent thoughts; it is not the *increment*. The action space is continuous.
-   *Policy* - a simple Gaussian policy $\pi$ centred at the current state $H$:
    \[
        A = H + \sigma^2 \varepsilon
    \quad\text{where}\quad
        \varepsilon \sim \mathcal N(9, I);
    \]
    the hyperparameter $\sigma^2$ controls exploration, and is decayed over time.
-   *Reward* - an intrinsic, confidence-based reward: the average of the top $k$ log probs, for a hyperparameter $k$.

The latent thoughts are updated through the policy gradient. The reward function chosen is non-differentiable, so REINFORCE is used instead of standard backprop:
$$\begin{aligned}
    J(H)
&=
    \mathbb E_{A \sim \pi(\cdot \mid H)}[R(A)],
\\
    \nabla_H J(H)
&=
    \mathbb E_{A \sim \pi(\cdot \mid H)}[R(A) \nabla_H \log \pi(A \mid H)].
\end{aligned}$$
But, $\log \pi(A \mid H) = - \tfrac12 \|A - H\|_2^2 / \sigma^2 + \text{const}$. Taking the gradient,
\[
    \nabla_H \log \pi(A \mid H)
=   (A - H) / \sigma^2
=   \varepsilon,
\]
writing $A = H + \sigma^2 \varepsilon$ as before. The authors then use a single sample to estimate the gradient:
\[
    \nabla_H J(H)
\approx
    R(H + \sigma^2 \varepsilon) \varepsilon,
\]
leading to a 'gradient-ascent' update of
\[
    H^{t+1}
:=  H^t + \eta R(H^t + \sigma^2 \varepsilon^t) \varepsilon,
\]
where $\eta > 0$ is the learning rate. To emphasise, this is a noisy estimate of gradient ascent.

After $T$ optimisation steps, the optimised latent thought vectors $H^\star$ are concatenated with the prompt embeddings $E(x)$ and passed through the AR LLM:
\[
    y
=   \operatorname{decode}\bigl(\mathcal M(E(x) \mathbin\Vert H^\star)\bigr).
\]

We point out that this gradient estimate uses no baseline. This will, no doubt, lead to very high variance estimators. It is a natural place to use GRPO.

>   Sample $\varepsilon_g \sim N(0, 1)$ and set $R_g := R(H + \varepsilon_g)$ for $g = 1, ..., G$, independently. The gradient update becomes
    \[\textstyle
        \nabla_H J(H)
    \approx
        \frac1G
        \sum_{g=1}^G
        \hat A_g \varepsilon_g
    \]
    where
    \[
        \hat A^g
    :=  \bigl( R_g - \operatorname{mean}(\{R_g\}_{g=1}^G) \bigr) / \operatorname{std}(\{R_g\}_{g=1}^G).
    \]

Each $R_g$ requires a forward pass through the frozen LLM, so can be batched. It's more compute per optimisation step, but likely far more efficient.

Currently, their method is framed as RL, but it's really just random search that just happens to use the REINFORCE estimator.


## Experiments

LTPO is compared against three baselines.

1.  **Zero-Shot CoT**
    - the standard, discrete-space CoT, instructing the model to generate explicit, step-by-step thinking. In a variant, untuned `[UNK]` tokens are appended: "[t]his baseline isolates the contribution of our test-time optimization procedure for latent thought tokens."
2.  **SoftCoT**
    performs reasoning in the continuous, latent space. It outperforms Coconut in certain cases, for example.
3.  **LatentSeek**
    applies test-time optimisation; unlike LTPO, it uses full AR decoding to evaluate intermediate steps.

![Performance of LTPO vs baselines](attachments/LTPO%20-%20Table%201.png)

Further experiments are conducted and comparisons in the §4 of the paper, not reported here, including the following.

-   Impact of number of thought tokens
-   Inference efficiency
-   (In)sensitivity to top-$k$ reward hyperparameter
-   Generalisation to other domains
-   Scalability with extended generation length


## Conclusion

The paper decouples generation of latent reasoning from the LLM - which was trained on discrete tokens. Instead, the latent thought vectors are optimised directly at test time, via RL.

Getting a good reward signal at test time is the real challenge. They use an internal confidence-based reward, in essence aiming to sharpen the distribution. This definition is somewhat ad hoc, and certainly leaves open research into a better reward.

Overall, the paper is written well, with the experiments clearly laid out.