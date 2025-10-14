---
tags:
    - hierarchy
    - latent reasoning
    - reasoning
    - RL
    - small model
    - training
method: TRM
title: 'Less is More: Recursive Reasoning with Tiny Networks'
lab: Samsung
date: 202510
---

# Tiny Recursive Model Summary

![TRM architecture](attachments/TRM%20-%20Overview.png){ align="right", width="50%" style="padding: 20px; padding-top: 0ex; padding-right: 0ex" }

-   [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/pdf/2510.04871)
-   2025-10; Jolicoeur-Martineau

[TOC]


## High-Level Summary

-   Introduces *Tiny Recursive Model* (TRM), a new small model, with hierarchy:
    -   high-level is current best guess to answer
    -   low-level does latent reasoning
-   Builds on, refines and improves ideas from [HRM](HRM.html), without focusing on biological analogies
-   Gets 45% accuracy on ARC-AGI-1 with only 7M parameters
-   Improvements are *largely* engineering based (imo):
    -   overall structure is the same
    -   high-/low-level networks use same parameters
    -   2-layer network repeated vs original 4-layer network
    -   stopping criterion is refined slightly
    -   a more memory-intensive recursion is used


## HRM Overview

Wang et al ([2025/06](https://arxiv.org/abs/2506.21734)) introduced the *Hierarchical Reasoning Model* (HRM) ([summary](HRM.html)). It is a supervised learning model with two novelties.

1.  *Recursive hierarchical reasoning*
    consists of recursing multiple times through two small networks: low-level $f_L$ at high frequency and high-level $f_H$ at low frequency, generating latent features $z_L$ and $z_H$, respectively. Biological motivation is provided.

2.  *Deep supervision*
    consists of improving the answer through multiple supervision steps, whilst carrying over the two latent features as initialisation for the next step.

Independent analysis by the ARC Prize team ([2025/08](https://arcprize.org/blog/hrm-analysis)) ([summary](HRM%20-%20ARC%20Analysis.html)) suggests performance on ARC-AGI is driven by the deep supervision, whilst the hierarchical structure only slightly improved performance.

The objective of the current work is to ("massively") improve the *recursive hierarchical reasoning*, leading to *Tiny Recursive Model* (TRM). TRM is also significantly smaller.

>   **NB.** There is notational clash between HRM and TRM, particularly in the pseudocode (Figures 2 and 3, respectively).
>
>   -   Number of high-/low-level recursions:
>
>       -   HRM uses $n$ high-level recursions, each of which consists of $T$ low-level recursions.
>       -   TRM uses $T$ high-level recursions, each of which consists of $n$ low-level recursions.
>
>       (Fortunately) by default, HRM uses $(T, n) = (2, 2)$, which is symmetric.
>
>   -   TRM interprets $z_H$ as the current output and $z_L$ as the latent reasoning, so (often, but not always) uses notation $y := z_H$ and $z := z_L$.
>
>   -   TRM uses $N_\textsf{sup}$ for the maximum number of supervision steps, whilst HRM uses $M_\textsf{max}$.


## From HRM to TRM

### Single, Shallower Network

HRM uses *two* networks: high-level $f_H$ and low-level $f_L$. TRM replaces these with a single network $f$, halving the parameter count in one fell swoop.

Increasing the number of layers is a natural way to (try to) increase the model capacity. Trying this, the TRM author found overfitting. On the contrary, *decreasing* the number of layers whilst scaling the number $n$ of recursions proportionally, they found 2 layers (instead of 4 in HRM) maximised generalisation. This halves the parameter count again, whilst (approximately) maintaining the emulated depth $n_\textsf{layers} (n+1) T M$.


### Depth of Hierarchical Structure

The HRM authors justify using *two* hierarchies via biological arguments, including comparisons with actual brain experiments on mice. No ablation over the number of layers is given. The TRM author gives a simpler explanation:

![Hierarchical layers ablation](attachments/TRM%20-%20Ablation%20-%20Num%20Features.png){ align="right", width="50%" style="padding: 10px; padding-top: 0ex; padding-right: 0ex" }

-   $y := z_H$ is simply the current (embedded) solution;
-   $z := z_L$ is a latent (reasoning) feature.

With this interpretation, there is no apparent reason for splitting the latent feature $z$ into multiple features. Nevertheless, an ablation table is provided. There, $n = 6$ high-level recursions are used.

The latent feature $z$ *can* be transformed into an (embedded) solution by $y \leftarrow f_H(x, y, z)$, but *shouldn't* be. Their Figure 6 highlights that $z_H$ does correspond to the (current) solution, whilst $z_L$ does not.


### Twice the Forward Passes with ACT

HRM uses *adaptive computational time* (ACT) during training to optimise the time spent on each data sample. It comes at a cost: the Q-learning objective requires an extra forward pass to estimate whether to stop now or continue. The current parameters are used, so this forward pass cannot be reused after the parameters have been updated.

TRM learns only a *halt* probability, through binary cross-entropy loss vs the correct solution. This removes the need for an (expensive) extra forward pass. No significant difference in generalisation was observed.


### Optimal Number of Recursions

![TRM depth ablation](attachments/TRM%20-%20Ablation%20-%20Depth.png){ align="right", width="50%" style="padding: 10px; padding-top: 0ex; padding-right: 0ex" }

An ablation study across varying values of $T$ and $n$ was conducted.

-   They found $(T, n) = (3, 6)$, equivalent to $n_\textsf{layers} (n+1) T = 2 \times (6 + 1) \times 3 = 48$ recursions optimal in TRM.
-   In contrast, HRM uses $(T, n) = (3, 3)$, equivalent to $n_\textsf{layers} (n+1) T = 4 \times (3 + 1) \times 3 = 42$ recursions.

TRM requires backpropagation through a *full* recursion process (see below). Thus, increasing $n$ too much leads to OOM errors.


### One-Step Gradient Approximation

HRM only backpropogates through the last two of the six recursions. This reduces the memory footprint significantly, and saves compute. It is justified via the *implicit function theorem*, which allows computation to the gradient *at a fixed point* in only one step.

The legitimacy of the *application* is questionable:

-   it requires *both* $z_L$ *and* $z_H$ to be at (or, near) the fixed point;
-   this is far from guaranteed when $(n, T) = (2, 2)$, as in their experiments.

TRM applies a full recursion process containing $n$ evaluations of $f_L$ and one of $f_H$ (now both $f$). Multiple backpropagation-free recursion processes can still be used to improve $(z_L, z_H) = (y, z)$.

An increase in generalisation on Sudoku is claimed, albeit without a proper ablation. This comes at the fairly significant cost of backpropping through the full process—see, eg, OOM errors in Table 3 when each latent recursion has more than 8 steps.


### Exponential Moving Average

On small data, (the TRM author claims that) HRM tends to overfit quickly and then diverge. An exponential moving average, a common technique in GANs and diffusion models used to improve stability, is integrated into TRM.


## Results

TRM is tested on puzzle benchmarks (Sudoku and mazes) and ARC-AGI (both 1 and 2, pass@2). The same data structure and augmentation is used as in HRM; see the [*Data Usage and Augmentation* section](HRM.html#data-usage-and-augmentation) in the [HRM summary](HRM.html) for details.

<p align="center">
    <img
        src="attachments/TRM - Results - Puzzles.png"
        width="49%"
        style="vertical-align: top; padding-right: 5px"
    >
    <img
        src="attachments/TRM - Results - ARC-AGI.png"
        width="49%"
        style="vertical-align: top; padding-left: 5px"
    >
</p>