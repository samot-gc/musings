---
tags:
    - DAPO
    - GRPO
    - PPO
    - TRPO
    - RL
    - training
    - own
method: RL
title: 'RL Algorithms: Overview for LLMs'
lab: OT
date: 202507
---

# RL Algorithms: Deep-Dive

-   *RL Algorithms: Deep-Dive*
-   2025-06; Olesker-Taylor

[TOC]


## High-Level Objective

The high-level objective for policy optimisation is to maximise the expected reward over policies:

\[
    \textstyle
\textsf{maximise}\quad
    \mathbb E_\pi\bigl[\sum_{t=0}^H R(s_t)\bigr]
\quad\textsf{over policies}\quad
    \pi;
\]

here, $(s_t, a_t)$ is the time-$t$ state–action pair, $R$ is the reward function and $H$ is some time horizon. Typically, the policies are parametrised by a vector $\theta$, and are *stochastic policies*: $\pi_\theta(a \mid s)$ is the probability of taking action $a$ when in state $s$.

This deep-dive is focused on policy optimisation, and has four parts.

1.  Vanilla policy gradient
2.  Introduce *trust regions*: TRPO
3.  Extend to *proximal policy optimisation* (v1 & v2)
4.  Introduce *group relative* estimation: GRPO



## Vanilla Policy Gradient

The 'vanilla' policy gradient algorithm uses a *baseline* $b$ to estimate the advantage. This could be a value function, perhaps being learned.

-   for iteration = 1, 2, ...:
    -   Collect a set of trajectories by executing the current policy
    -   At each timestep $t$ in each trajectory, compute
        -   the *return* $R_t = \sum_{t' \ge t} \gamma^{t' - t} r(s_{t'}, a_{t'}, s_{t'+1})$
        -   the *advantage estimate* $\hat A_t = R_t - b(s_t)$
    -   Re-fit the baseline by minimising $(b(s_t) - R_t)^2$, summed over all trajectories and timesteps, eg via Monte Carlo estimates or boostrapping
    -   Update the policy using a policy gradient estimate, which is a sum of all terms $\nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat A_t$

See [Lecture 3: *Policy Gradients and Advantage Estimation*](https://www.youtube.com/watch?v=AKbX1Zvo7r8&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0&index=3&t=2115s) in Pieter Abbeel's [*Foundations of Deep RL* series](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0) for an overview or [Lecture 5: *Policy Gradients*](https://www.youtube.com/watch?v=GKoKNYaBvM0&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=15) in Sergey Levine's [*Deep RL* course](https\://rail.eecs.berkeley.edu/deeprlcourse/) for a detailed account.


## TRPO: Trust Region Policy Optimisation

[TRPO](http://arxiv.org/abs/1502.05477) was introduced by Schulman et al in 2015; see also Schulman's [thesis](http://joschu.net/docs/thesis.pdf). The 'trust region' constrains how far away the new policy can be from the current one. In doing so, it enables some local approximations which make the optimisation easier.

Again, an overview is given in Pieter Abbeel's [Lecture 4: *TRPO and PPO*](https://www.youtube.com/watch?v=KjWF8VIMGiY&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0&index=4) and a detailed account in Servey Levine's [Lecture 9: *Advanced Policy Gradients*](https://www.youtube.com/watch?v=ySenCHPsKJU&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=36).

### Preliminaries

Before proceeding formally, we set up some notation. For simplicity, we assume an infinite time horizon: $H = \infty$. We assume that the initial state $S_0 \sim p_0$ and there is some transition kernel $p$ such that $S_{t+1} \sim p(\cdot \mid s, a)$ given $(S_t, A_t) = (s, a)$, *independent of the policy*.

Given a policy $\pi$, denote the *state–action value function*

\[
    Q_\pi(s, a)
\textstyle
=   \mathbb E_{(S_t, A_t)_{t\ge0} \sim \pi}\bigl[ \sum_{t\ge0} \gamma^t r(S_t, A_t, S_{t+1}) \bigm| (S_0, A_0) = (s, a) \bigr],
\]

the expected discounted (future) reward given the current state is $s$ and action $a$ is taken. Then, the *value function*

\[
    V_\pi(s)
=   \mathbb E_{A \sim \pi(\cdot \mid s)}[ Q_\pi(s, A) ]
\]

is the average value wrt $\pi$. The *advantage* wrt $\pi$ quantifies the benefit of choosing action $a$ over following $\pi$ on average:

\[
\begin{aligned}
    \mathcal A_\pi(s, a)
&=  Q_\pi(s, a) - V_\pi(s) \\
&=  Q_\pi(s, a) - \mathbb E_{A \sim \pi(\cdot \mid s)}[ Q_\pi(s, a) ]
=   \mathbb E_{A \sim \pi(\cdot \mid s)}[ Q_\pi(s, a) - Q_\pi(s, A) ].
\end{aligned}
\]

and the *expected discounted reward* is

\[
\begin{aligned}
    R(\pi)
=   \mathbb E_{S \sim p_0}[ V_\pi(S) ]
=   \mathbb E_{S \sim p_0, A \sim \pi(\cdot \mid S)}[ Q_\pi(S, A) ]
\textstyle
=   \mathbb E_{(S_t, A_t)_{t\ge0} \sim \pi}[ \sum_{t\ge0} \gamma^t r(S_t, A_t) ].
\end{aligned}
\]

### First-Order Simplification

We can use the 'nested' nature of the sums to obtain an expression for the change in expected reward when changing from policy $\pi$ to $\pi'$:

\[
    \textstyle
    R(\pi') - R(\pi)
=   \mathbb E_{(S'_t, A'_t)_{t\ge0} \sim \pi'}\bigl[ \sum_{t\ge0} \gamma^t A_\pi(S'_t, A'_t) \bigr].
\]

Indeed, the state–action value function satisfies

\[
    Q_\pi(s, a)
=   \mathbb E_{S \sim p(\cdot \mid s, a)}[ r(s, a, S) + \gamma V_\pi(S) ],
\]

where $p(\cdot \mid s, a)$ denotes the one-step probabilities from state $s$ choosing action $a$. In particular, if $(S'_t, A'_t)_{t\ge0} \sim \pi'$, then $S'_{t+1} \sim p(\cdot \mid S'_t, A'_t)$ given $(S'_t, A'_t)$. Hence,

\[
\begin{aligned}
    \textstyle
    \mathbb E_{\pi'}\bigl[ \sum_{t\ge0} \gamma^t \mathcal A_\pi(S'_t, A'_t) \bigr]
&=  \textstyle
    \mathbb E_{\pi'}\bigl[ \sum_{t\ge0} \gamma^t \bigl( r(S'_t, A'_t, S'_{t+1}) + \gamma V_\pi(S'_{t+1}) - V_\pi(S'_t) \bigr) \bigr] \\
&=  \textstyle
    \mathbb E_{\pi'}\bigl[ \sum_{t\ge0} \gamma^t r(S'_t, A'_t, S'_{t+1}) - V_\pi(S'_0) \bigr]
=   R(\pi') - R(\pi),
\end{aligned}
\]

by the telescoping nature of the sum, recalling that $S_0, S_0' \sim p_0$—ie, for both $\pi$ and $\pi'$.

We can rewrite this equation as a sum over *states*, rather than *timesteps*. To this end, for a policy $\pi$, let

\[
    \textstyle
    \rho_\pi(s)
=   \sum_{t\ge0}
    \gamma^t
    \mathbb P_\pi(S_t = s),
\]

the discounted occupation measure under $\pi$; to emphasise $S \sim \rho_\pi$ means $\mathbb P(S = s) \propto \rho_\pi(s)$. Note that

\[
    \textstyle
    \sum_s
    \rho_\pi(s)
=   \sum_{t\ge0}
    \gamma^t
=   1/(1 - \gamma).
\]

Then, we can write the increment 

\[
\begin{aligned}
    R(\pi') - R(\pi)
&\textstyle
=   \sum_{t\ge0}
    \sum_{s'}
    \mathbb P_{\pi'}(S'_t = s')
    \sum_{a'}
    \pi'(a' \mid s')
    \mathcal A_\pi(s', a')
\\&\textstyle
=   \sum_{s'}
    \rho_{\pi'}(s')
    \sum_{a'}
    \pi'(a' \mid s')
    \mathcal A_\pi(s', a')
\\&\textstyle
=   \sum_{s'}
    \rho_{\pi'}(s')
    \mathbb E_{A' \sim \pi'(\cdot \mid s')}[ \mathcal A_\pi(s', A') ]
\\&\textstyle
=   \tfrac1{1-\gamma}
    \mathbb E_{S' \sim \rho_{\pi'}, A' \sim \pi'(\cdot \mid S')}[ \mathcal A_\pi(S', A') ];
\end{aligned}
\]

in other words, sample a state $S' \sim \rho_{\pi'}$ according to the $\pi'$-occupation measure $\rho_{\pi'}$ and an action $A' \sim \pi'(\cdot \mid S')$ for this state according to $\pi'$, then take the expectation.

The sum $\sum_{a'} \pi'(a' \mid s) \mathcal A_\pi(s', a')$ is a relatively simple function of $\pi'$. Eg, if $\pi' = (1 - \varepsilon) \pi + \varepsilon \nu$, then

\[
    \textstyle
    \sum_{a'}
    \pi'(a' \mid s')
    \mathcal A_\pi(s', a')
=   \varepsilon
    \sum_{a'}
    \nu(a' \mid s')
    \mathcal A_\pi(s', a'),
\]

since $\mathbb E_{A \sim \nu(\cdot \mid s)}[\mathcal A_\nu(s, a)] = 0$ for any $\nu$—"the advantage of a measure over itself is $0$". However, the dependence of $\rho_{\pi'}(s')$ on $\pi'$ is far more complex.

We plan to do small updates $\pi \to \pi'$, by gradient ascent, or similar. As such, we simply replace $\rho_{\pi'}$ by $\rho_\pi$:

\[
    \widetilde R_\pi(\pi')
=   R(\pi) + \tfrac1{1-\gamma} \Delta_\pi(\pi')
\]

where

\[
    \textstyle
    \Delta_\pi(\pi')
:=  (1 - \gamma)
    \sum_s \rho_\pi(s)
    \mathbb E_{A' \sim \pi'(\cdot \mid s)}[ \mathcal A_\pi(s, A') ]
=   \mathbb E_{S \sim \rho_\pi, A' \sim \pi'(\cdot \mid s)}[ \mathcal A_\pi(S, A') ].
\]

We now parametrise the policy $\pi$ by a vector $\theta$, and overload notation; eg, $R_\theta(\theta') = R_{\pi_\theta}(\pi_{\theta'})$. Then,

\[
    \widetilde R_{\theta_0}(\theta_0) = R_{\theta_0}(\theta_0)
\quad\textsf{and}\quad
    \nabla_\theta \widetilde R_{\theta_0}(\theta) |_{\theta = \theta_0} = \nabla R(\theta) |_{\theta = \theta_0}.
\]

Indeed, $\rho_{\pi_\theta} \to \rho_{\pi_{\theta'}}$ does not change the leading order as $\theta' \to \theta$, and it already appears in the 'derivative' (difference) part. So, a sufficiently small step $\pi_{\theta} \to \pi_{\theta'}$ that improves $\widetilde R_\theta = \widetilde R_{\pi_\theta}$ also improves the original $R$. But, it does not give any guidance on how large a step can be taken.

### Constrained Optimisation

One potential avenue is to use a *conservative mixture*: let $\pi_\star = \mathop{\textsf{arg}}\mathop{\textsf{max}}_{\pi_\star} L_\pi(\pi_\star)$ and define $\pi' := \alpha \pi_\star + (1 - \alpha) \pi$. Then, [Kakade & Langford (2002)](https://dl.acm.org/doi/10.5555/645531.656005) derived the lower bound

\[
    \textstyle
    R(\widetilde \pi) - \widetilde R_\pi(\widetilde \pi)
\ge - 2 \varepsilon \gamma (1 - \gamma)^{-2} \alpha^2
\quad\textsf{where}\quad
    \varepsilon
=   \mathop{\textsf{max}}_s | \mathbb E_{A' \sim \pi'(\cdot \mid s)}[\mathcal A_\pi(s, A')] |.
\]

This policy class is unwieldy and restrictive in practice. In fact, the proof can be extended to show, for any policy $\pi'$, that

\[
    R(\pi') - \widetilde R(\pi')
\ge - 4 \varepsilon' \gamma (1 - \gamma)^{-2} \mathop{\textsf{KL}}^\star(\pi \mathrel{\|} \pi')
\quad\textsf{where}\quad
    \textstyle
    \mathop{\textsf{KL}}^\star(\pi \mathrel{\|} \pi')
=   \mathop{\textsf{max}}_s
    \mathop{\textsf{KL}}\bigl( \pi(\cdot \mid s) \bigm\| \pi'(\cdot \mid s ) \bigr).
\]

This lower bound on $R(\pi') - \widetilde R(\pi')$ tends to be pessimistic in practice. An alternative approach, inspired by this, is to use a constrained optimisation problem:
\[
    \textsf{given $\pi$},
\quad
    \textsf{maximise}
\quad
    L_\pi(\pi')
\quad
    \textsf{over policies $\pi'$}
\quad
    \textsf{subject to}
\quad
    \mathop{\textsf{KL}}^\star(\pi \mathrel{\|} \pi')
\le \delta.
\]

However, this imposes a KL-constraint at every state $s$, making it impractical. Instead, we replace this by an *average* KL-divergence:

\[
    \overline{\mathop{\textsf{KL}}}(\pi \mathrel{\|} \pi')
=   \mathbb E_{S \sim \rho_\pi}\bigl[ \mathop{\textsf{KL}}\bigl( \pi(\cdot \mid S) \bigm\| \pi'(\cdot \mid S ) \bigr) \bigr].
\]

It turns out (see below) that this average is much easier to estimate. We use it in our constraint:

\[
    \textsf{given $\pi$},
\quad
    \textsf{maximise}
\quad
    L_\pi(\pi')
\quad
    \textsf{over policies $\pi'$}
\quad
    \textsf{subject to}
\quad
    \overline{\mathop{\textsf{KL}}}(\pi \mathrel{\|} \pi')
\le \delta.
\]

Such a constrained optimisation problem can be (approximately) solved efficiently using a conjugate gradient; see [Appendix C *Efficiently Solving the Trust-Region Constrained Optimization Problem*](https://arxiv.org/pdf/1502.05477#appendix.C) from the original [TRPO paper](https://arxiv.org/pdf/1502.05477) or the [*Solving KL Penalized Problem*](https://youtu.be/xvRrgxcpaHY?t=1233&si=DKqTSA913DgeDrZK) chapter from John Schulman's lecture on policy gradients, TRPO and PPO for more details.

### Importance Sampling

Using importance sampling, we can replace a law $\pi'$ by $\pi$:

\[
    \mathbb E_{A' \sim \pi'(\cdot \mid s)}[ \mathcal A_\pi(s, A') ]
=   \mathbb E_{A  \sim \pi (\cdot \mid s)}\bigl[ \tfrac{\pi'(A \mid s)}{\pi(A \mid s)} \mathcal A_\pi(s, A) \bigr].
\]

Let's parametrise the policies, overloading notation:

\[
    \Delta_\theta(\theta' \mid s)
:=  \Delta_{\pi_\theta}(\pi_{\theta'} \mid s)
:=  \mathbb E_{A \sim \pi_\theta(\cdot \mid s)}\bigl[ \tfrac{\pi_{\theta'}(A \mid s)}{\pi_\theta(A \mid s)} \mathcal A_{\pi_\theta}(s, A) \bigr];
\]

then, by the chain rule,

\[
    \nabla_{\theta'} \Delta_\theta(\theta' \mid s) |_{\theta' = \theta}
=   \mathbb E_{A \sim \pi_\theta(\cdot \mid s)}\bigl[ \nabla_\theta \log \pi_\theta(A \mid s) \mathcal A_{\pi_\theta}(s, A) \bigr].
\]

So, the gradient at the current policy is the same as in the usual policy gradient.

### Sample-Based Estimation of the Objective and Constraint

The importance sampling also means that we need only *evaluate* $\pi'(\cdot \mid s)$ at a (random) location $A$, not *sample* from $\pi'(\cdot \mid s)$. Instead, we just need to sample $S \sim \rho_\pi$ and then $A \sim \pi(\cdot \mid S)$ given $S$. This enables us to estimate the expectation of

\[
    \widetilde R_\pi(\pi')
=   R(\pi) +
    \tfrac1{1-\gamma}
    \mathbb E_{S \sim \rho_\pi, A \sim \pi(\cdot \mid S)}\bigl[ \tfrac{\pi'(A \mid S)}{\pi(A \mid S)} \mathcal A_\pi(S, A) \bigr],
\]

by simply averaging $(s, a) \mapsto \widetilde{\mathcal A}_{\pi, \pi'}(s, a) = \tfrac{\pi'(a \mid s)}{\pi(a \mid s)} \mathcal A_\pi(s, a)$ over rollouts $(S^{(i)}_t, A^{(i)}_t)_{t\ge0} \sim^\mathsf{iid} \pi$ ($i = 1, ..., M$, say) under the initial measure $\pi$. This even enables evaluation of *multiple* candidate $\pi'$-s from *the same* collection of rollouts. When optimising over $\pi'$, we may ignore the additive $R(\pi)$ term.

We can estimate the KL term in the constraint via similar rollouts. A naive method samples many actions per (observed) state to estimate the KL divergence $\mathop{\textsf{KL}}\bigl( \pi(\cdot \mid s) \bigm\| \pi'(\cdot \mid s ) \bigr)$ directly. A smarter method expands the KL:

\[
    \textstyle
    \overline{\mathop{\textsf{KL}}}(\pi \mathrel{\|} \pi')
=   \mathbb E_{S \sim \rho_\pi}\bigl[ \sum_a \pi(a \mid S) \log \frac{\pi'(a \mid S)}{\pi(a \mid S)} \bigr]
=   \mathbb E_{S \sim \rho_\pi, A \sim \pi(\cdot \mid S)}\bigl[ \log \frac{\pi'(A \mid S)}{\pi(A \mid S)} \bigr].
\]

This can be estimated via rollouts under $\pi$ as before. In fact, we likely even collected the ratios $\frac{\pi'(A \mid S)}{\pi(A \mid S)}$ during the estimation of the objective.

Last, we could actually use a different distribution in the importance sampling. Using $A \sim \pi(\cdot \mid s)$ is perhaps the simplest option, and referred to as *single path* in Section 5.1 of the original [TRPO paper](https://arxiv.org/pdf/1502.05477); a *vine* method, which requires being able to restore the system to particular previous states, is discussed in Section 5.2 there.

### Using Gradient Ascent and the Fisher Information

Alternatively, the KL can be estimated more 'directly' using the Fisher information matrix $F$:

\[
    \overline{\mathop{\textsf{KL}}}(\pi_{\theta'} \mathrel{\|} \pi_\theta)
\approx
    \tfrac12 (\theta' - \theta)^\top F (\theta' - \theta)
\quad\textsf{where}\quad
    F = \mathbb E_{\pi_\theta}[ \nabla_\theta \log \pi_\theta(A \mid S) \nabla_\theta \log \pi_\theta(A \mid S)^\top ].
\]

The Fisher matrix is just an expectation, so can be estimated with (the same) samples. This leading-order approximation can be combined with a linearisation of the objective:

\[
    \textsf{replace}
\quad
    \Delta_\theta(\theta')
=   \mathbb E_{S \sim \rho_{\pi_\theta}, A \sim \pi_\theta(\cdot \mid S)}\bigl[ \tfrac{\pi_{\theta'}(A \mid S)}{\pi_\theta(A \mid S)} \mathcal A_{\pi_\theta}(S, A) \bigr]
\quad\textsf{with}\quad
    \nabla_{\theta'} \Delta_\theta(\theta')^\top (\theta' - \theta),
\]

and recall that $\nabla_{\theta'} \Delta_\theta(\theta') |_{\theta' = \theta} = \nabla_\theta R(\theta) |_{\theta' = \theta}$. The resulting constrained gradient ascent problem, namely

\[
    \textsf{maximise}
\quad
    \nabla_{\theta'} \Delta_\theta(\theta')^\top (\theta' - \theta)
\quad
    \textsf{over parameters $\theta'$}
\quad
    \textsf{subject to}
\quad
    \tfrac12 (\theta' - \theta)^\top F (\theta' - \theta)
\le \delta
\]

can then be solved exactly:

\[
    \theta'
=   \theta + \alpha F^{-1} \nabla_\theta R(\theta)
\quad\textsf{where}\quad
    \alpha
=   \sqrt{2 \delta / \nabla_\theta R(\theta)^\top F \nabla_\theta R(\theta) }.
\]

This is detailed in Sergey Levine's [Lecture 9, Part 4](https://www.youtube.com/watch?v=QWnpF0FaKL4&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=39).


### Estimated Advantage

We may not have access to the advantage $\mathcal A_\pi(s, a)$, and need to replace it with an estimator $\widehat{\mathcal A}_\pi(s, a)$. There are a couple of main options.

| Method | Bias | Variance | Sample Efficiency |
|-|-|-|-|
| MC | Low | High | Low |
| A3C | High | Low | High |
| GAE | Tunable | Tunable | Balanced |

The time-$t$ reward is

\[
    G
=   r_0 + \gamma r_1 + \gamma^2 r_2 + \ldots.
\]

*Monte Carlo* (MC) simply averages rollouts, using the full trajectory: $G^\textsf{MC} = G$; this makes it slow. Also, the rewards may vary significantly in space, leading to a high-variance estimate.

*Asynchronous Advantage Actor–Critic* (A3C) goes in the other direction:

\[
    G^{\textsf{A3C}(k)}
=   r_0 + \gamma r_1 + \ldots + \gamma^k r_k + \gamma^{k+1} V(s_{k+1}),
\]

where $s_{k+1}$ is the time-$(k+1)$ state and $V$ is the value function—which itself may be being learnt. An estimator of this has reduced variance, since it is only looking a few steps into the future.

Finally, *Generalised Advantage Estimation* (GAE) takes an exponential weighting of the $\textsf{A3C}(k)$:

\[
    \textstyle
    G^{\textsf{GAE}(\lambda)}
=   (1 - \lambda)
    \sum_{k\ge0}
    \lambda^k G^{\textsf{A3C}(k)}
= \ldots =
    \sum_{k\ge0}
    (\gamma \lambda)^k
    \bigl( r_k + \gamma (1-\lambda) V(s_{k+1}) \bigr).
\]

Equivalently, $G^{\textsf{GAE}(\lambda)} = \mathbb E_{K \sim \mathop{\textsf{Geom}}_0(\lambda)}[ G^{\textsf{A3C}(K)} ]$. This interpolates between A3C(0) ($\lambda = 0$) and Monte Carlo ($\lambda = 1$).

## PPO: Proximal Policy Optimisation

[PPO](https://arxiv.org/abs/1707.06347) moves the KL condition from the constraint to the optimisation.

-   TRPO: maximise $\mathbb E_{\pi'}\bigl[ \frac{\pi'(A \mid S)}{\pi(A \mid S)} \mathcal A_\pi(S, A) \bigr]$ subject to $\overline{\mathop{\textsf{KL}}}(\pi \mathrel{\|} \pi') \le \delta$
-   PPO: maximise $\mathbb E_{\pi'}\bigl[ \frac{\pi'(A \mid S)}{\pi(A \mid S)} \mathcal A_\pi(S, A) \bigr] - \beta \overline{\mathop{\textsf{KL}}}(\pi \mathrel{\|} \pi')$

The method of Lagrange multipliers actually says that, given $\delta$, there is some $\beta$ so that the optimality points agree (ie, same optimiser).

In practice, it can be difficult to choose the hyperparameter $\beta$. One adaptive option uses *dual descent updates* for $\beta$, to try to match it to the original constraint $\delta$. Iterate 

-   **while** unsatisfied: find the optimal $\pi'_\beta$ for the current $\beta$, and abbreviate $\overline d := \overline{\mathop{\textsf{KL}}}(\pi \mathrel{\|} \pi'_\beta)$
    -   If $\overline d$ is too *large* (say, $\overline d > \tfrac32 \delta$), then *increase* the penalty $\beta$ (say, $\beta \leftarrow 2\beta$)
    -   If $\overline d$ is too *small* (say, $\overline d < \tfrac23 \delta$), then *decrease* the penalty $\beta$ (say, $\beta \leftarrow \beta/2$)
    -   Otherwise, the KL is 'good enough' (say, $\tfrac23 \delta \le \overline d \le \tfrac32 \delta$), so become satisfied

See Sergey Levine's [Lecture 9, Part 3](https://www.youtube.com/watch?v=WuPauZgX7BM&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=38) for more details.

A popular extension adjusts the objective by *clipping* the probability ratio $r_\pi(\pi') := \pi'(a \mid s) / \pi(a \mid s)$, hiding the dependence on $(s, a)$:

\[
    \textsf{previously,}
\quad
    \mathbb E_\pi[ r_\pi(\pi') \mathcal A_\pi ];
\qquad
    \textsf{now,}
\quad
    \mathbb E[ \mathop{\textsf{min}}\{ r_\pi(\pi') \mathcal A_\pi, \mathop{\textsf{clip}}(r_\pi(\pi'), 1 - \varepsilon, 1 + \varepsilon) \mathcal A_\pi \} ].
\]

The clipping stops the probability ratio $r_\pi(\pi')$ from changing too much. The objective is the pessimistic minimum of this clipping and the previous, unclipped version. The clipping has no effect to first order around the current policy—ie, if $\pi' \approx \pi$. It is really there to stop overly large updates from happening, which was observed without this clipped term in the minimum.

Thinking about positive advantage for the moment, we want to increase the probability ratio. Once the clipping $1 + \varepsilon$ is achieved, changing $\pi'$ cannot increase the ratio. So, there is nothing to be gained beyond this point. An analogous statement holds for pushing the ratio down when the advantage is negative. This restricts the influence for each state–action pair.

This tends to work a bit better than TRPO on continuous control (eg, robotics) and much better on discrete problems (eg, Atari).


## GRPO: Group Relative Policy Optimisation

### General Method

[GRPO](http://arxiv.org/abs/2402.03300) removes the need for a critic model, used to estimate the value of actions and frequently resource-expensive. Instead , a "group" of actions is sampled, and their *relative* rewards are compared. This does not take into account future rewards (ie, the value function at the new state), which the critic in PPO does. Further, it is an *off-policy* method: there is a fixed distribution $\mu$ over states (or questions) from which the states are drawn.

More precisely, instead of drawing a single action $A \sim \pi(\cdot \mid s)$, many (iid) actions $A_1, ..., A_G \sim^\mathsf{iid} \pi(\cdot \mid s)$ are drawn. The $i$-th advantage $\widehat{\mathcal A}_i$ (ie, of $A_i$) is calculated *relative to $(A_1, ..., A_G)$*. For example, one option is to take

\[
    \widehat{\mathcal A}_i = (r_i - \mathop{\textsf{mean}} r) / \mathop{\textsf{std}} r,
\]

where $r_i$ is the (random) reward received after taking action $A_i$; naturally,

\[
    \textstyle
    \mathop{\textsf{mean}} r = \frac1G \sum_{i=1}^G r_i
\quad\textsf{and}\quad
    \mathop{\textsf{std}} r = \sqrt{ \frac1G \sum_{i=1}^G (r_i - \mathop{\textsf{mean}} r) }.
\]

Additionally, a KL penalisation term is included in the objective, between the new policy $\pi'$ and a *reference* policy $\pi_\textsf{ref}$. This *is not* the old policy $\pi_\theta$; DeepSeek suggest it is usually the initial supervised fine-tune model—ie, the one *before* GRPO RL is applied.

Altogether, the final surrogate objective, to be maximised, is

\[
    \textstyle
    L^\textsf{GRPO}_\pi(\pi')
=   \mathbb E_{S \sim \mu, A_1, ..., A_G \sim^\mathsf{iid} \pi(\cdot \mid S)}\bigl[
        \tfrac1G
        \sum_{i=1}^G
        \bigl(
            \mathop{\textsf{min--clip}}_\varepsilon^\varepsilon(r_i,  \widehat{\mathcal A}_i)
        -   \beta \mathop{\textsf{KL}}(\pi' \mathrel{\|} \pi_\textsf{ref})
        \bigr)
    \bigr]
\]

where
\(
    r_i
:=  \pi'(A \mid S) / \pi(A \mid S)
\)
and $\widehat{\mathcal A}_i$ is the (relative) advantage, as before; here,
\[
    \mathop{\textsf{min--clip}}_\varepsilon^\varepsilon(r, a)
:=  \min\{r a, \mathop{\textsf{clip}}(r, 1 + \varepsilon, 1 - \varepsilon) a\}
=   \min\{ r a, (1 + \operatorname{sgn}(a) \varepsilon) a\}.
\]
Importantly, the *whole* sum must be inside the expectation, since the (relative) advantage is calculated based on $(A_1, ..., A_G)$. Notice how $S \sim \mu$, not a measure which depends on the current policy $\pi$ (or the proposed $\pi'$). The clipping is a type of regularisation, preventing the ratio from changing too much: there is no insentive to go beyond $1 \pm \varepsilon$.

We (unnecessarily) left the KL inside the expectation, and even the summation. This is because, rather than calculate the KL exactly, the following unbiased estimator is typically used:

\[
    \textstyle
    \widehat{\mathop{\textsf{KL}}}_i
:=  r^\textsf{ref}_i - 1 - \log r^\textsf{ref}_i
\ge 0
\quad\textsf{where}\quad
    r^\textsf{ref}_i
:=  \pi_\textsf{ref}(A_i \mid S) / \pi(A_i \mid S).
\]

The clipping prevents the new policy $\pi'$ from being too far from the old policy $\pi$, and this final KL penalty prevents *any* policy used from being too far from the *original* (reference) policy $\pi_\textsf{ref}$.

Noteably, GRPO *does not* consider any future rewards—it would need a critic (or some other addition) to do that. This makes is suitable for Question–Answer models, which is the context it was introduced in. Such scenarios are special because there are no 'onward states' beyond the answer, and hence no future rewards.


### Set-Up for Question–Answer Models

Question–Answer models can, and frequently do, have *reasoning traces* (interpreted broadly). In particular, given a question $q$, an action $A$ may generate output $O_1$, ..., $O_m$, where $m$ is the (enforced) length of the reasoning. In this case, and individual reward *can* (but needn't) be applied to each part of the trace. This leads to a summation of the form

\[
    \textstyle
    \tfrac1G
    \sum_{i=1}^G
    \tfrac1{|O_i|}
    \sum_{t=1}^m
    \mathop{\textsf{min--clip}}_\varepsilon^\varepsilon(r_{i,t}, \widehat{\mathcal A}_{i,t})
\]

where
\(
    r_{i,t}
=   \pi'(o_{i,t} \mid q, o_{i, 1:t-1}) / \pi(o_{i,t} \mid q, o_{i, 1:t-1})
\)
and $\widehat{\mathcal A}_{i,t}$ is the advantage of the $t$-th token in the $i$-th trace. A canonical choice is to reward the token based on the final result (there is no need for the reward to be 'Markovian'):
\(
    \widehat{\mathcal A}_{i,t}
=   \widehat{\mathcal A}_t.
\)
But, it is possible to use more refined strategies.


### Training Differences: TRPO/PPO vs GRPO

One important distinction between the TRPO/PPO and GRPO paradigms is the manner in which the (surrogate) objective is estimated.

-   TRPO/PPO is an *on-policy* method.
    -   It samples a long trajectory $(S_t, A_t)_{t\ge0}$ in a given step, in order to get a good estimate on the expectation, which includes $S \sim \rho_\pi$, which takes into account the discounting.
    -   If the environment does not permit long samples (>1k steps), such as an Atari game, then many trajectories are sampled; all start from a fixed distribution $p_0$ over initial states.
    -   As the algorithm processes, the distribution $\rho_\pi$ changes.

-   GRPO is an *off-policy* method.
    -   The state (eg, question) is drawn from a *fixed* distribution $S \sim \mu$. Once the state $S$ has been sampled, a group of $G$ actions are sampled *from the same state*: $A_1, ..., A_G \sim^\mathsf{iid} \pi(\cdot \mid S)$.
    -   These $G$ state–action pairs (each with the same state) are averaged.

GRPO really is designed with question–answer models in mind. TRPO/PPO make less sense in this context, since there are no future rewards to discount (a bit like $\gamma = 0$), and the trajectories are all single-step.


## DAPO: Decoupled Clip and Dynamic Sampling Policy Optimisation

[DAPO](http://arxiv.org/abs/2503.14476) introduces five main adjustments to the GRPO algorithm. Evaluation of the algorithm can be found in their [paper](https://arxiv.org/pdf/2503.14476). This summary is restricted to the algorithmic aspects. The training code, built on [`verl`](https://github.com/volcengine/verl), along with the curated and processed is open-sourced; see the [project page](https://dapo-sia.github.io/) (https://dapo-sia.github.io/).

<!--
1.  *Clipping higher* promotes diversity of the system and avoids entropy collapse.
2.  *Dynamic sampling* improves training efficiency and stability.
3.  *Token-level loss* is critical in long CoT RL scenarios.
4.  *Overlong reward shaping* reduces reward noise and stabilises training.
5.  *Removing the KL penalty* allows divergence from the reference without the KL penalty dominating.
-->

### Adjustments

1.  <details>
    <summary><b>Clip Higher</b>: promotes diversity of the system and avoids entropy collapse.</summary>

    When the entropy of the policy decreases quickly as training progresses, the model prematurely commits to a narrow set of outputs, limiting exploration. By increasing the upper bound in the clipping, low-likelihood, high-reward tokens get boosted significantly more, removing some constraint on exploration. We write

    \[
        \mathop{\textsf{min--clip}}_{\varepsilon_\textsf{low}}^{\varepsilon_\textsf{high}}(x)
    :=  \min\{x, \mathop{\textsf{clip}}(x, 1 - \varepsilon_\textsf{low}, 1 + \varepsilon_\textsf{high})\}.
    \]

    The lower value $\varepsilon_\textsf{low}$ is usually left at the default $0.2$. The upper value $\varepsilon_\textsf{high}$ is taken to be $\approx 0.25$ in [*Magistral*](http://arxiv.org/abs/2506.10910) and $0.4$ in [*ProRL*](https://arxiv.org/abs/2505.24864).
    </details>


2.  <details>
    <summary><b>Dynamic Sampling</b>: improves training efficiency and stability.</summary>

    If all outputs $o_1, ..., o_G$ are correct/incorrect, then the resulting advantage is zero. This dilutes the batch gradients. DAPO proposes filtering out prompts with accuracy equal to $1$ or $0$, oversampling until the batch is complete.

    The sampling cost for each batch is dynamic, due to the unknown amount of filtering needed. The claim is that this does not necessarily impede training efficiency, because the generation time is typically dominated by the generation of long-tail samples.

    An alternative is to filter the prompts, then upweight the remainder. Eg, if a proportion $\tfrac14$ are filtered out, the remainder are upweighted by $\tfrac43 = 1/(1 - \tfrac14)$.
    </details>


3.  <details>
    <summary><b>Token-Level Policy Gradient Loss</b>: critical in long CoT RL scenarios.</summary>

    The original GRPO employs a *sample-level* loss calculation: first, average the losses by token within each sample; then, aggregate the losses across samples. Each *sample* is thus assigned an equal weight. As such, tokens in longer responses are underweighted relatively. This can lead to two main adverse effects.

    1.  Learning reasoning-relevant patterns in long samples can be impeded.
    2.  Long samples may exhibit low-quality patterns (eg, gibberish or repetition), which dominate the high-value tokens.
    
    <br>
    DAPO addresses this via a *token-level* loss calculation: each *token* is assigned the same weight. Notationally,

    \[
        \textstyle
        |G|^{-1} \sum_{i=1}^G |o_i|^{-1} \sum_{t=1}^{o_i} (...)
    \quad\textsf{is replaced with}\quad
        (\sum_{i=1}^G |o_i|)^{-1} (...).
    \]
    </details>


4.  <details>
    <summary><b>Overlong Reward Shaping</b>: reduces reward noise and stabilises training.</summary>

    A maximum length for generation is usually set, with overlong samples truncatd. This can introduce reward noise, as sound reasoning may be penalised due to its excessive length, and disrupt training significantly. DAPO first applies an *overlong filtering* strategy, masking the loss of truncated samples. Further, it uses a soft punishment:

    -   no penalty when the length is at most $L_\textsf{max} - L_\textsf{cache}$;
    -   maximal penalty when the length exceeds $L_\textsf{max}$;
    -   linear interpolated penalty between these thresholds.
    
    <br>
    </details>


5.  <details>
    <summary><b>Remove KL Penalty</b>: allows divergence without KL penalty dominating.</summary>

    THe KL penalty term $-\beta \mathop{\mathsf{KL}}(\pi' \mathrel{\|} \pi_\textsf{ref})$ regulates divergence between the online policy and the frozen reference policy. However, during training the long-CoT reasoning model, the policy can diverge significantly, leading to the KL term dominating the loss. For this reason, DAPO removes the KL term.

    [ProRL](http://arxiv.org/abs/2505.24864) takes a more nuonced approach: the KL term remains, but the reference policy is reset/updated periodically.
    </details>


### Ablation-Type Study

To assess the contribution of each aspect, they are added to the base model, DeepSeek-R1-Zero-Qwen-32B, one at a time—except the KL penalty, which is never included. The model is evaluated on AIME'24 using avg@32.

| Model                   | AIME'24 |
| ----------------------- | ------- |
| *DeepSeek base*         | **47%** |
| *Naive GRPO*            | **30%** |
| + overlong filtering    | 36%     |
| + clip-higher           | 38%     |
| + soft overlong penalty | 41%     |
| + token-level loss      | 42%     |
| + dynamic sampling      | **50%** |


## Potential Extension for GRPO

The lack of a critic model makes the process both faster to train, but also (often, more importantly) more memory efficient. The critic may often be as large as the modle being trained. However, it is my feeling that it is really the *lack of future rewards* which removes the need for a critic, rather than the use of *relative rewards*:

-   estimating value functions is difficult because of the need to behave optimally in the future;
-   if the future rewards are ignored ($\gamma = 0$), this goes away.

The sample means and standard deviations are proxies for the true means and standard deviations under the current policy. These may be quite crude estimators when the number $G$ of samples is small—eg, $G = 4$. Potentially, a better estimate on these—eg, by using a large number of samples—may improve performance.

However, increasing $G$ may hurt performance elsewhere: the (surrogate) objective is an average, so the signal may decrease, by some type of concentration/LLN. One option could be to increase $G$ and divide by $\sqrt G$ instead of $G$.
