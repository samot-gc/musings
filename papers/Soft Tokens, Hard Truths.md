---
tags:
    - continuous thoughts
    - inference
    - latent
    - reasoning
    - rl
    - training
method: FlowRL
parent: 'FlowRL: Matching Reward Distributions for LLM Reasoning'
authors:
    - Meta FAIR
date: 202509
---

# Soft Tokens, Hard Truths Summary

-   [Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170)
-   2025-09; Butt, Kwiatkowski, Labiad, Kempe, Ollivier

[TOC]


## High-Level Summary

-   Introduces a scalable method to learn continuous CoTs via RL
-   Uses 'soft' tokens: mixtures of tokens together with noise on the input embeddings
-   Matches discrete-token CoT for pass@1 metrics and surpasses for pass@32, suggesting greater CoT diversity
-   Interestingly, the best-performing scenarios involve *training* with continuous CoT tokens but using discrete-tokens for *inference*
-   Better out-of-domain evaluations, suggesting a 'softer touch'
-   Tons of benchmark comparisons, leading to an excellent paper; shame the code isn't provided...


## Elevator Pitch

Standard CoT is constrained by the discreteness of language tokens and the sequential nature of the sampling. Contrastingly, human cognition often operates over abstract and fluid concepts, rather than linguistic symbols. This motivates enabling LLMs to reason in *continuous concept spaces*.

Instead of sampling a token according to a (softmax) distribution, [*Soft Thinking*](Soft%20Thinking.html) simply takes the weighted average of all tokens. *Soft Tokens, Hard Truths* takes this a step further by adding noise—an essential component for (standard) RL training. This adds minimal computational overhead unlike, for example, [*COCONUT*](COCONUT.html).

![Transformer structure](attachments/Soft%20Tokens,%20Hard%20Truths%20-%20Structure.png){ style="display: block; margin: 0 auto" }

*Contributions and Findings*:
-   a scalable continuous-token post-training algorithm;
-   parity for pass@1 and improvements for pass@32;
-   improved robustness and out-of-distribution performance;
-   "hard inference on soft models".


## Methodology

In a nutshell, average wrt the full distribution after the softmax step, instead of sampling, then add noise.


### Notation for LLM Architecture

-   A *token* is a one-hot (row) vector $x \in \mathbb R^{1 \times V}$.
-   The *token embedding matrix* $E \in \mathbb R^{V \times n_0}$ takes a sequence (stack) $x_{< t} \in \mathbb R^{t \times V}$ of tokens and returns a sequence $h_{< t}^0 := x_{< t} E \in \mathbb R^{t \times n_0}$ of embeddings.
-   The *transformer stack* $\mathcal T$ turns the sequence $h_{< t}^0 \in \mathbb R^{t \times n_0}$ of input embeddings into a sequence $h_{< t}^L := \mathcal T(h_{< t}^0) \in \mathbb R^{t \times n_L}$ of output embeddings.
-   The output encodings $h_{< t}^L$ are turned into *logits* by a *decoding matrix* $W \in \mathbb R^{n_L \times V}$. The logits are turned into *next-token probabilities* $p_{< t}$ are obtained by a softmax: $p_{< t} := \operatorname{softmax}(h_{< t}^L W / \tau) \in \mathbb R^{t \times V}$ with $\tau \ge 0$.


### Standard (Hard) Tokens

In standard, hard-token models, the next token $x_t$ is sampled according to the last component of $p_{< t}$:
\[
    \Pr(x_t = 1_v)
=   p_{t - 1, v},
\]
where $1_v \in \mathbb R^V$ is the one-hot encoding of token $v \in V$. This is applied inductively to get the sequences of next tokens.


### Soft Thinking

In *Soft Thinking*, during the CoT phase, instead of *sampling* the next token according to $p_{t-1}$, the probabilities are used to define a mixture of embeddings, which is used as the next input layer:
\[
\textstyle
    h_t^0
:=  \sum_v
    p_{t - 1, v} e_i
=   p_{t - 1} E
\in \mathbb R^{1 \times n_0}
\]
where $e_v = 1_v E \in \mathbb R^{1 \times V}$, and $p_{t - 1} \in \mathbb R^{1 \times V}$ and $E \in \mathbb R^{V \times n_0}$.

Once the CoT phase is complete, the model samples normal (hard) tokens.

This model is not amenable to direct RL training via REINFORCE-like algorithms, since there is no underlying randomness. In principle, one could backprop through all timesteps of the CoT, but this leads to technical memory challenges not discussed in the paper.

[*Mixture of Inputs*](Mixture%20of%20Inputs.html) mixes a *hard* (sampled) token and the *soft* mixture, introducing randomness. The choice of mixture seemed pretty application-specific. The authors of that paper did not try RL training.


### Noisy Soft Thinking: Soft Tokens and Fuzzy Tokens

The current paper proposes simply adding noise to the soft mixture, to make soft thinking RL trainable:
\[
    \tilde h_t^0
:=  p_{t-1} E + \sigma N(0, 1);
\]
at the next timestep, the transformer stack is fed $\tilde h_t^0$.

They call this model *soft tokens*. When the CoT temperature $\tau \approx 0$, they use the term *fuzzy tokens*, since then the non-noisy embeddings $h^0_t = p_{t-1} E$ reduce to (greedily chosen) discrete tokens.


### RL on Soft Tokens

The randomness means that the usual RL algorithms, such as RLOO, GRPO, DAPO and so on, can now be applied in the usual way. Some details are given at the end of §3.


## Experiments

There are many possible variants: soft tokens during training, hard sampling during inferences, trained on GSM8K and evaluated on MATH-500; etc. Laudably, many variations are run and compared—each taking up to four days!


### Training–Inference Set-ups

-   *Training*
    -   *Hard tokens*: categorical sampling of ordinary hard CoT tokens with temp $\tau = 1.0$
    -   *Soft tokens*: full mixture at temp $\tau = 0.5$, plus Gaussian noise
    -   *Fuzzy tokens*: like soft tokens, but at temp $\tau = 0.00001$ (almost greedy + noise)
    -   *Noise scale*: $\sigma = \tfrac13 \mathsf{RMSN}$ where $\mathsf{RMSN} := \mathsf{RMS}(\{\| E_{v, \cdot} \|\}_{v \in V})$ is the RMS of the norms of the token embeddings $E_{v, \cdot} \in \mathbb R^{n_0}$
-   *Inference*
    -   *Hard Greedy*: discrete tokens, CoT temp $\tau = 0.0$ (greedy)
    -   *Hard Sample*: discrete tokens, CoT temp $\tau = 1.0$
    -   *Soft Greedy*: no noise (scale $\sigma = 0$), CoT temp $\tau = 0.5$
    -   *Soft Sample*: noise scale $\sigma = \tfrac13 \mathsf{RMSN}$, CoT temp $\tau = 0.5$
    -   *Fuzzy Greedy*: no noise (scale $\sigma = 0$), CoT temp $\tau = 0.00001$ (almost *hard greedy*)
    -   *Fuzzy Sample*: noise scale $\sigma = \tfrac13 \mathsf{RMSN}$, CoT temp = $0.00001$


### Datasets and Base Models

...
