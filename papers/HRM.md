---
tags:
    - hierarchy
    - latent reasoning
    - reasoning
    - RL
    - small model
    - training
method: HRM
title: 'Hierarchical Reasoning Model'
lab: Sapient
date: 202506
---

# Hierarchical Reasoning Model Summary

-   [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
-   2025-06; Wang, Li, Sun, Chen, Liu, Wu, Lu, Song, Yadkori

[TOC]


## High-Level Summary

-   Introduces *Hierarchical Reasoning Model* (HRM), a new small model
-   A recurrent architecture modelled on hierarchical and multi-timescale processing in human brains
-   Attains significant computational depth whilst maintaining both training stability and efficiency
-   Small model: only 27M parameters
-   operators without pre-training: task-specific training


## Elevator Pitch

Current LLMs primarily employ CoT techniques to reason. These suffer from computational challenges, including requiring extensive data. Also, are they *really* reasoning? Or, are they recalling reasoning they've seen? Their struggles on novel tasks, such as in ARC, hint at the latter.

![HRM architecture](attachments/HRM%20-%20Headline%20Architecture.png){ align="right", width="40%" style="padding: 20px" }

Inspired by hierarchical processing and temporal separation in the human brain, HRM has two interdependent recurrent modules:

-   a high-level module responsible for slow, abstract planning;
-   a low-level module handling rapid, detailed computations.

The model operates without pre-training of CoT data. Rather, training data is required for the task at hand—eg, ARC-AGI-2 or Maze-Hard. This checkpoint is not saved for future tasks: the model is randomly initialised for each task.

HRM outperforms much larger models with significantly longer context windows on ARC, and achieves excellent performance on complex tasks, such as Sudoku or optimal path finding in large mazes, where CoT models fail completely.

![HRM results](attachments/HRM%20-%20Headline%20Results.png){ style="display: block; margin: 0 auto" }


## Detailed Motivation

LLMs largely rely on CoT prompting for reasoning. CoT *externalises* reasoning into language, breaking down complex tasks into simpler, intermediate steps. Such a process lacks robustness and is tethered to linguistic patterns; it requites significant training data and generates many tokens, resulting in large computational cost. Humans certain don't reason this way.

Enter *latent reasoning*: the model *internalises* computations. This aligns with the understanding that language is *tool* for communication, not the *substrate* of thought itself. Latent reasoning is fundamentally constrained by a model's *effective computational depth*. Naively stacking layers leads to vanishing gradients. Recurrent architectures often suffer from early convergences—internal state stabilises, and further calculations are useless, and rely on computationally expensive *backprop through time* (BPTT) training.

The human brain organises computation hierarchically across different regions, operating at different timescales. Slow, higher-level areas guide fast, lower-level circuits via recurrent feedback loops. This increases the effective depth, without the cost of BPTT.

HRM is designed to significantly increase the effective computational depth. It features two coupled recurrent modules:

![HRM modules' interaction](attachments/HRM%20-%20Modules%20Interaction.png){ align="right", width="60%" style="padding-left: 10px" }

-   high-level for slow, abstract reasoning;
-   low-level for fast, detailed computations.

The slow module (H) advances only after the fast one (L) has reached a local equilibrium, at which point it is reset. This avoids the early convergence of standard recurrent models.

Further, a one-step gradient approximation for training HRM is used, maintaining $O(1)$ memory footprint (vs $O(T)$ for $T$ timesteps in BPTT), applying backprop in a single step at the equilibrium point.


## HRM: The Nuts and Bolts

The HRM consists of four learnable components:

-   an input network $f_I(\cdot; \theta_I)$;
-   a low-level recurrent module $f_L(\cdot; \theta_L)$;
-   a high-level recurrent module $f_H(\cdot; \theta_H)$;
-   an output network $f_O(\cdot; \theta_O)$.

The models' dynamics unfold over $N$ high-level cycles, each consisting of $T$ low-level steps. The modules $f_{L/H}$ each keep a hidden state $z_{L/H}$.

The HRM maps an input vector $x$ to an output prediction vector $\hat y$ as follows.

1.  Convert the input $x$ into a working representation $\tilde x$:
    \[
        \tilde x
    :=  f_I(x; \theta_I).
    \]

2.  At each timestep $i = 1, ..., NT$, the L module updates its state and the H module updates only if a multiple of $T$ as follows:
    \[
    \begin{aligned}
        z_L^i
    &:=
        f_L(z_L^{i-1}, z_H^{i-1}, \tilde x; \theta_L);
    \\
        z_H^{i/T}
    &:=
        f_H(z_H^{i-1}, z_L^{i-1}; \theta_H)
    \quad\text{if}\quad
        i \equiv 0 \text{ mod } T.
    \end{aligned}
    \]

3.  After $N$ full cycles, a prediction $\hat y$ is extracted from the hidden state of the H module:
    \[
        \hat y
    :=  f_O(z_H^{NT}; \theta_O).
    \]

Pseudocode of HRM with deep supervision training in PyTorch is given below.

```python
def hrm(z, x, N=2, T=2):
    x = input_embedding(x)
    zH, zL = z

    # iterate modules without gradient
    with torch.no_grad():
        # iterate L-level module
        for i in range(N * T - 1):
            zL = L_net(zL, zH, x)
            # iterate H-level module
            if (i + 1) % T == 0:
                zH = H_net(zH, zL)
    
    # one-step grad
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)
    return (zH, zL), output_head(zH)
```

This entire three-step process represents a single forward pass. A halting mechanism (discussed below) determines whether the model should terminate, in which case $\hat y$ is output, or continue with an additional forward pass.

Whilst convergence is crucial for recurrent networks, it also limits their *effective depth*—iterations the (almost) convergence point add little value. HRM addresses this by alternating between L-convergence and H-iteration: a single H update establishes a fresh context for the L module, 'restarting' its convergence.

<details style="margin: 2em">
<summary><i>Theoretical foundation for one-step gradient approximation.</i></summary>

The one-step gradient approximation is based on *deep equilibrium models*, which employ the *implicit function theorem* to bypass BPTT.

In idealised HRM behaviour, during the $k$-th high-level cycle, in state $z_H^{k-1}$, the L module iterates its state $z_L$ to a local fixed point $z_L^\star$. This can be expressed in the form
\[
    z_L^\star
=   f_L(z_L^\star, z_H^{k-1}, \tilde x; \theta_L),
\]
where $\tilde x = f_I(x; \theta_I)$. The H module then performs a single update using this converged state:
\[
    z_H^k
=   f_H(z_H^{k-1}, z_L^\star; \theta_H),
\]
In other words,
\[
    z_H^k
=   f(z_H^{k-1}; \theta, \tilde x)
\]
for some $f$, where $\theta = (\theta_I, \theta_L, \theta_H)$. Its fixed point $z_H^\star = z_H^\star(\theta, \tilde x)$ satisfies
\[
    z_H^\star
=   f(z_H^\star; \theta).
\]
The implicit function theorem allows calculation of the exact gradient *of the fixed point* $z_H^\star$ wrt $\theta$ through the Jacobian $J_F(z_H) := \frac{\partial F}{\partial z_H}$:
\[
    \frac{\partial z_H^\star}{\partial \theta}
=   \bigl( I - J_F(z_H^\star) \bigr)^{-1}
\cdot
    \frac{\partial F}{\partial \theta} \biggr|_{z_H = z_H^\star}.
\]

Unfortunately, evaluating and inverting $I - J_F(z_H)$ is prohibitively expensive. But, for $z_H$ in a neighbourhood of the fixed point, $J_F(z_H) \approx 0$. This leads to
\[
    \frac{\partial z_H^\star}{\partial \theta_H}
\approx
    \frac{\partial f_H}{\partial \theta_H},
\quad
    \frac{\partial z_H^\star}{\partial \theta_L}
\approx
    \frac{\partial f_H}{\partial z_L^\star}
\cdot
    \frac{\partial z_L^\star}{\partial \theta_L}
\quad\textsf{and}\quad
    \frac{\partial z_H^\star}{\partial \theta_I}
\approx
    \frac{\partial f_H}{\partial z_L^\star}
\cdot
    \frac{\partial z_L^\star}{\partial \theta_I}.
\]
The gradients of the low-level fixed point $z_L^\star$ can be approximated similarly:
\[
    \frac{\partial z_L^\star}{\partial \theta_L}
\approx
    \frac{\partial f_L}{\partial \theta_L}
\quad\textsf{and}\quad
    \frac{\partial z_L^\star}{\partial \theta_I}
\approx
    \frac{\partial f_L}{\partial \theta_I}.
\]
Plugging these into the previous display gives the final approximated gradients:
\[
    \frac{\partial z_H^\star}{\partial \theta_H}
\approx
    \frac{\partial f_H}{\partial \theta_H},
\quad
    \frac{\partial z_H^\star}{\partial \theta_L}
\approx
    \frac{\partial f_H}{\partial z_L^\star}
\cdot
    \frac{\partial f_L}{\partial \theta_L}
\quad\textsf{and}\quad
    \frac{\partial z_H^\star}{\partial \theta_I}
\approx
    \frac{\partial f_H}{\partial z_L^\star}
\cdot
    \frac{\partial f_L}{\partial \theta_I};
\]
these depend only on the functions $f_{H/L/I}$.
</details>


<details open style="margin: 2em">
<summary><i>Halting strategy and loss function.</i></summary>

The adaptive halting strategy uses Q-learning to adaptively determine the number of segments (forward passes). A Q-head uses the final state of the H module to predict the Q-values:
\[
    \hat Q^m
=   (\hat Q^m_\textsf{half}, \hat Q^m_\textsf{cont})
=   \operatorname{sigmoid}(\theta_Q^\top z_H^{m N T}).
\]
The half/cont(inue) action is chosen using a randomised strategy.

-   Let $M_\textsf{max}$ denote the maximum number of segments (a fixed hyperparameter).
-   Let $M_\textsf{min} := 1 + \operatorname{Bern}(\varepsilon) \operatorname{Unif}(\{0, ..., M_\textsf{max} - 1\})$ denote the minimum number of segments.
-   The halt action is selected if $m \ge M_\textsf{min}$ and either $m \ge M_\textsf{max}$ or $\hat Q_\textsf{half} > \hat Q_\textsf{cont}$.

The Q-head is updated through a Q-learning algorithm. The state of its episodic *Markov decision process* (MDP) at segment $m$ is $\tilde z^m := z_H^{m N T}$ and the action space is $\{\textsf{halt}, \textsf{cont}\}$.

-   Choosing $\textsf{halt}$ terminates the episode and returns the binary reward indicating correctness: $\mathbf 1\{ \hat y^m = y \}$.
-   Choosing $\textsf{cont}$ yields a reward of $0$ and the state transitions to $\tilde z^{m+1}$.

The Bellman equations define the optimum:
\[
\begin{aligned}
    Q^\star_\textsf{halt}(\tilde z)
&=  \mathbb E[ \mathbf 1\{ \hat y^m = y \} ];
\\
    Q^\star_\textsf{cont}(\tilde z)
&=  \mathbb E[ \max\{Q^\star_\textsf{halt}(\tilde z), Q^\star_\textsf{cont}(\tilde z)\} ].
\end{aligned}
\]

The optimum $Q^\star_\textsf{halt/cont}(\tilde z)$ is interpreted as the probability of being correct given halting/continuing, given that the current state is $\tilde z$.

Thus, the Q-learning *targets* $\hat G^m = (\hat G^m_\textsf{halt}, \hat G^m_\textsf{cont})$ for the two actions are given as follows:
\[
\begin{aligned}
    \hat G^m_\textsf{halt}
&:=
    \mathbf 1\{ \hat y^m = y \};
\\
    \hat G^m_\textsf{cont}
&:=
\begin{cases}
    \hat Q^{m+1}_\textsf{halt}
&\text{if}\quad m \ge M_\textsf{max},
\\
    \max\{ \hat Q^{m+1}_\textsf{halt}, \hat Q^{m+1}_\textsf{cont} \}
&\text{otherwise}
\end{cases}
\end{aligned}
\]
This last expression corresponds to the probability of being correct given continuing at step $m$: either stop at step $m+1$ ($\hat G^{m+1}_\textsf{halt}$) or continue beyond ($\hat G^{m+1}_\textsf{cont}$).

We can now define our loss function:
\[
    \mathcal L^m
=   \textsf{Loss}(\hat y^m, y)
+   \textsf{BinaryCrossEntropy}(\hat Q^m, \hat G^m).
\]
Minimising this enables both accurate predictions and sensible stopping decisions.
</details>

[[..up to part way through page 8...]]