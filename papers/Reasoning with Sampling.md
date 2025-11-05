---
tags:
    - inference
    - reasoning
    - RL
    - training-free
method: Harvard
title: 'Reasoning with Sampling: Your Base Model is Smarter Than You Think'
lab: Meta
date: 202510
---

# Reasoning with Sampling: Your Base Model is Smarter Than You Think

-   [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901)
-   2025-10; Karan, Du
-   GitHub: https://github.com/aakaran/reasoning-with-sampling/tree/main

[TOC]


## High-Level Summary

-   Capability vs Sampling:
    -   There is significant debate/interest in whether RL post-training *develops new capabilities* or just *sharpens sampling*
    -   If it's the latter, perhaps the distribution can be sharpened without any post-training
-   Paper's contribution:
    -   Uses MCMC to sample from sharpened power distributions ($p^\alpha$ instead of $p$, for some $\alpha \ge 1$)
    -   Find substantial gains over base model and comparable with RL-post-trained versions
    -   No training, datasets, verifiers or the like are needed


## Elevator Pitch

Post-training an LLM with RL often provides impressive improvements for pass@1, but pass@k decays for large $k$. This raises the question, "Does RL develop new capabilities, or simply sharpen the distribution?" Eg, it may collapse to high-reward modes.

The current work achieves similar pass@1 performance increase *by pure sampling, without RL*. Moreover, pass@k remains competitive with the base even for $k$ up to $16$.

Training LLMs to *reason* incentivises them to output their *chain of thought* (CoT). This allows exploration of different strategies, and backtracking, but comes at a significant computational cost: their CoTs can be *very long*, inflating the context length by an order of magnitude and more. This is expensive, both in terms of compute and latency, since it is serialised, not parallelised.

![Base vs GRPO vs theirs](attachments/Reasoning%20with%20Sampling%20-%20Figure%201.png)

*Contributions and Findings*:

-   sharpening distributions can match RL post-training
-   sharpening can require expensive test-time compute
-   performance boost appears to persist for larger pass@k


## Methodology for Sharpening

*Sharpening* a distribution corresponds to reweighting it so that high-likelihood regions becomes even higher, whilst low-likelihood become even lower.


### Power Distributions

The authors utilise *power distributions*:

>   given a distribution $p$ and real $\alpha$, the *power distribution* $p^\alpha$ is defined such that $p^\alpha(x) \propto p(x)^\alpha$ for all $x$.

Importantly, this is different to changing the temperature of the LLM sampler:
\[\begin{aligned}
    p_{\textsf{pow}(\alpha)}(x_t \mid x_{< t})
&\textstyle
\propto
    \sum_{x_{> t}}
    p(x_{< t}, x_t, x_{> t})^\alpha,
\\
    p_{\textsf{temp}(\alpha)}(x_t \mid x_{< t})
&\textstyle
\propto
    \bigl( 
        \sum_{x_{> t}}
        p(x_{< t}, x_t, x_{> t})
    \bigr)^\alpha,
\end{aligned}\]
where $p_{\textsf{pow}(\alpha)} = p^\alpha$ is the $\alpha$-power distribution and /$p_{\textsf{temp}(\alpha)}$ is the $1/1\alpha$-temperature distribution.

Intuitively, low-temperature sampling affects only the current token: it does not account for the likelihood of "future paths". Conversely, the power distribution up-weights the *entire* path. Naturally, sampling from $p^\alpha$ exactly is computationally intractable—even calculating the normalising constant is. Instead, a Metropolis–Hastings algorithm is used.


### Metropolis–Hastings

The authors use a standard Metropolis–Hastings algorithm. This draws approximate samples from a target distribution $\pi$ given only a proposal distribution $q$. It is iterative:

-   if $x_t = x$, draw $y \sim q(\cdot \mid x)$;
-   accept $y$, setting $x_{t+1} := y$, with probability $A(x, y)$ where
    \[
        A(x, y)
    :=  \min\biggl\{ 
            1, \:
            \frac{ q(y \mid x) \pi(x) }{ q(x \mid y) \pi(y) }
        \biggr\};
    \]
-   otherwise, reject $y$, setting $x_{t+1} := x$.

Being able to evaluate $\pi(\cdot)$ is not necessary, only calculating ratios is; in particular, an unnormalised version can be used instead. Practically, both $q(x \mid y)$ and $q(y \mid x)$ should be easily computable—or, at least, their ratio.

The target distribution in the current set-up is $\pi := p_\textsf{LLM}^\alpha$; the choice of $q$ is open. The following process is used:

>   given sequence $x = (x_0, ..., x_T)$, choose $L \sim \operatorname{Unif}(\{1, ..., T\})$ and resample the sequence starting at index $L$ using (the usual) $p_\textsf{LLM}$.

The transition probabilities $q(y \mid x)$ and $q(x \mid y)$ are then simple to calculate.