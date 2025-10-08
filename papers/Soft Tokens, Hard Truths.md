---
tags:
    - continuous thoughts
    - inference
    - latent
    - reasoning
    - rl
    - training
method: soft tokens
parent: 'Soft Tokens, Hard Truths'
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


### Configurations

Each configuration was run with 3 independent random seeds; the tablets report the resulting mean and standard deviation.

-   *Training* settings

    1.  *Hard tokens*: categorical sampling of ordinary hard CoT tokens with temp $\tau = 1.0$
    2.  *Soft tokens*: full mixture at temp $\tau = 0.5$, plus Gaussian noise
    3.  *Fuzzy tokens*: like soft tokens, but at temp $\tau = 0.00001$ (almost greedy + noise)
    -   *Noise scale*: $\sigma = \tfrac13 \mathsf{RMSN}$ where $\mathsf{RMSN}$ is the RMS of the norms of the token embeddings $E_{v, \cdot} \in \mathbb R^{n_0}$
    -   Train for 4k steps for each dataset (which is a lot!)

-   *Inference* settings

    1.  *Hard Greedy*: discrete tokens, CoT temp $\tau = 0.0$ (greedy)
    2.  *Hard Sample*: discrete tokens, CoT temp $\tau = 1.0$
    3.  *Soft Greedy*: no noise (scale $\sigma = 0$), CoT temp $\tau = 0.5$
    4.  *Soft Sample*: noise scale $\sigma = \tfrac13 \mathsf{RMSN}$, CoT temp $\tau = 0.5$
    5.  *Fuzzy Greedy*: no noise (scale $\sigma = 0$), CoT temp $\tau = 0.00001$ (almost *hard greedy*)
    6.  *Fuzzy Sample*: noise scale $\sigma = \tfrac13 \mathsf{RMSN}$, CoT temp = $0.00001$

-   *Base models*

    1.  Llama 3.2 3B Instruct
    2.  Llama 3.1 8B Instruct
    3.  Qwen 2.5 3B Instruct

-   *Datasets*

    -   Training (maths):
        1.  GSM8k
        2.  MATH
        3.  DeepScaleR
    -   Evaluation (maths):
        1.  GSM8K
        2.  MATH (specifically, subset MATH-500)
        3.  OlympiadBench (specifically, 675 questions with final answers and not containing images or figures)
    -   Evaluation (out-of-distribution):
        1.  HellaSwag
        2.  MMLU
        3.  ARC/AI2


### Results

So many comparisons are undertaken that it would be too much to report them here—and that's *highly* commendable. Only some of those in the body are repeated here, but the reader is encouraged to view the full paper.

Across datasets and models, the three training schemes are broadly comparable for greedy pass@1. This demonstrates that fuzzy and soft thinking are at least effective. Further, they have a clear overall advantage for pass@32 over hard training.

For all training settings, hard inference generally performs the best, both for pass@1 and pass@32. In particular, previous reported benefits of soft inference on hard (normal) training is not recovered.

![Hard inference - all models](attachments/Soft%20Tokens,%20Hard%20Truths%20-%20Hard%20Inference%20(all%20models).png){ style="display: block; margin: 0 auto" }

The authors point out one particular set-up:

>   Llama-8B-Instruct, trained on GSM8K and evaluated on MATH-500, only achieves good scores when soft/fuzzy CoT training is used; classical hard-token CoT is ineffective.

Base Llama-8B-Instruct (no fine-tuning) has good performance on GSM8K (presumably due to exposure in training), but this does not translate to good performances on MATH. Hard fine-tuning makes things worse (on MATH), but soft/fuzzy improve. This *suggests* that soft/fuzzy training bring more generalisation on Llama-8B-Instruct.

There is typically a gap between *hard greedy* ($\tau = 0.0$) and *hard sample* ($\tau = 1.0$) inference settings for the base models and models trained with soft/fuzzy CoT, whereas the gap with hard CoT training is small. This is highlighted by Figure 3.

![Hard inference - Llama](attachments/Soft%20Tokens,%20Hard%20Truths%20-%20Hard%20Inference%20(Llama).png){ style="display: block; margin: 0 auto" }


### Potential Red Flag

Figure 3 does raise a red flag: by the time k reaches 32, the base model (no fine-tuning) is basically indistinguishable on pass@k from the models fine-tuned with soft/fuzzy CoT (hard CoT is worse).

This gives evidences towards the idea that RL fine-tuning focuses the distribution to improve *sampling* (pass@k for small k) but not *capability* (large k). Certainly, at least for some of the plots, it *looks like* the orange line is steepers, even at k = 32.

Certainly, soft/fuzzy CoT training *appear* to have less of a negative impact for large k. However, as much as the authors hint otherwise, on *sample* pass@1, hard CoT systematically beats soft/fuzzy. That said, they are close for *greedy* pass@1. This *suggests* that soft/fuzzy CoT is perhaps somewhere between the two (ie, the base and hard CoT)?