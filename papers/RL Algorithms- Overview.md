---
tags:
    - dapo
    - grpo
    - ppo
    - trpo
    - rl
    - training
    - own
method: RL
parent: 'RL Algorithms: Overview for LLMs'
authors: Olesker-Taylor
date: 202506

---
# RL Algorithms: Overview for LLMs

-   *RL Algorithms: Overview for LLMs*
-   2025-06; Olesker-Taylor

[TOC]


## Framework and Objective

Given a question $q$, an LLM produces an output $o$. This output may consist of multiple steps ($|o| > 1$), of which the last is the actual answer; or, it may simply be the answer ($|o| = 1$). We consider the set-up of *reinforcement learning with verifiable rewards* (RLVR); this includes mathematical questions, or coding questions for which unit tests are provided. Contrast this with balancing a pole on a cart. In the LLM framework, a policy $\pi = \pi_\theta$ is indexed by the weights $\theta$ of the LLM.

The high-level objective is to maximise the expected reward:
\[
    \text{maximise}
\quad
    J(\theta)
:=  \mathbb E_{q \sim \mu, o \sim \pi_\theta(\cdot \mid q)}[ R(q, o) ]
\quad\text{over weights}\quad
    \theta,
\]
where $\mu$ is some *fixed* distribution over the questions. The reward $R(q, o)$ may be binary: $1$ if the answer is correct; $0$ if it is wrong. But, typically, it is more nuanced, and will depend on the full output $o$—including the reasoning trace.


## Policy Optimisation

This overview is intended to give an overview of the most important details. Further (less importoant) details can be found in my [RL deep-dive](RL%20Algorithms-%20Deep-Dive.html); the framework there is general state–action, not question–answer. My (older) [DeepSeekMath summary](DeepSeekMath%20GRPO.html) introduces GRPO in the QA framework.

I found some lectures/courses useful.

-   Sergey Levine's [Deep RL course](https://rail.eecs.berkeley.edu/deeprlcourse/) is clear and in-depth; see, particularly, Lectures 5 and 9 on *(Advanced) Policy Gradients*
-   Pieter Abbeel's [Foundations of Deep RL](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0) provides a decent overview, but the details are often fuzzy; see, particularly, Lectures 3 and 4 on *Policy Gradients* and *TRPO & PPO*.
-   John Schulman covers policy gradients, TRPO and PPO in a single [Deep RL Bootcamp lecture](https://www.youtube.com/watch?v=xvRrgxcpaHY)—but somewhat runs out of time when covering clipping.

### Vanilla Policy Gradients

In order to do a gradient step, we need to determine
\[\begin{aligned}
    J'(\theta)
&\textstyle
=   \mathbb E_{q \sim \mu}\bigl[
        \sum_o
        R(q, o)
        \nabla_\theta \pi_\theta(o \mid q)
    \bigr]
\\&\textstyle
=   \mathbb E_{q \sim \mu}\bigl[
        \sum_o
        R(q, o)
        \pi_\theta(o \mid q)
        \nabla_\theta \log \pi_\theta(o \mid q)
    \bigr]
\\&\textstyle
=   \mathbb E_{q \sim \mu, o \sim \pi_\theta(\cdot \mid q)}\bigl[
        R(q, o)
        \nabla_\theta \log \pi_\theta(q \mid o)
    \bigr],
\end{aligned}\]
using the identity
\[
    \nabla_\theta \log \pi_\theta(q \mid o)
=   \nabla_\theta \pi_\theta(q \mid o) / \pi_\theta(q \mid o).
\]
An alternative representation of the objective uses *importance sampling* (in essence, just change of measure):
\[
    J(\theta)
=   \mathbb E_{q \sim \mu, o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}\biggl[
        \frac{\pi_\theta(o \mid q)}{\pi_{\theta_\text{old}}(o \mid q)}
        R(q, o)
    \biggr].
\]
This removes the variable $\theta$ from the *generation* aspect: all generations come from the current policy $\pi_{\theta_\text{old}}$.

To reduce bias, the reward is usually replaced with the *advantage* $A^{\pi_{\theta_\text{old}}}(q, o)$ under the current policy:
\[
    A^\pi(q, o)
:=  R(q, o) - V^\pi(q)
\quad\text{where}\quad
    V^\pi(q)
:=  \mathbb E_{o \sim \pi(\cdot \mid q)}[ R(q, o) ].
\]
With this definition,
\[\begin{aligned}
    J(\theta) - J(\theta_\text{old})
&
=   \mathbb E_{q \sim \mu}\bigl[
        \mathbb E_{o \sim \pi_\theta(\cdot \mid q)}[R(q, o)]
    -   \mathbb E_{o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}[R(q, o)]
    \bigr]
\\&
=   \mathbb E_{q \sim \mu}\bigl[
        \mathbb E_{o \sim \pi_\theta(\cdot \mid q)}\bigl[
            R(q, o) - \mathbb E_{o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}[R(q, o)]
        \bigr]
    \bigr]
=   \mathbb E_{q \sim \mu, o \sim \pi_\theta(\cdot \mid q)}\bigl[ A^\pi(q, o) \bigr].
\end{aligned}\]
Applying the importance sampling trick,
\[
    J(\theta) - J(\theta_\text{old})
=   \Delta(\theta, \theta_\text{old})
:=  \mathbb E_{q \sim \pi, o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}\biggl[
        \frac{\pi_\theta(o \mid q)}{\pi_{\theta_\text{old}}(o \mid q)}
        A^{\pi_{\theta_\text{old}}}(q, o)
    \biggr].
\]
Given that the current policy is $\pi_{\theta_\text{old}}$, maximising either $\theta \mapsto J(\theta)$ or $\theta \mapsto \Delta(\theta, \theta_\text{old})$ is equivalent.
Below, $\Delta'$ indicates derivative ($\nabla_\theta$) with respect to the *first* argument; in particular, $\Delta'(\theta_\text{old}; \theta_\text{old}) = J'(\theta_\text{old})$.
<!--
Observe that
\[\begin{aligned}
    \Delta'(\theta_\text{old}; \theta_\text{old})
&
=   \mathbb E_{q \sim \pi, o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}\biggl[
        \frac{\nabla_\theta \pi_\theta(o \mid q) |_{\theta = \theta_\text{old}}}{\pi_{\theta_\text{old}}(o \mid q)}
        A^{\pi_{\theta_\text{old}}}(q, o)
    \biggr]
\\&
=   \mathbb E_{q \sim \pi, o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}\biggl[
        \nabla_\theta \log \pi_\theta(o \mid q) |_{\theta = \theta_\text{old}}
        A^{\pi_{\theta_\text{old}}}(q, o)
    \biggr]
=   J'(\theta_\text{old}).
\end{aligned}\]
-->

### TRPO: Trusted Region Policy Optimisation

In the usual RL framework, an entire trajectory of states (and actions) is sampled from the *proposed* policy. In the state–action framework, sampling instead from the *old* policy—which is far preferable computationally, as the same rollouts can be used to compare multiple proposals—leads to a first-order approximation; this is detailed in my [RL deep-dive](RL%20Algorithms-%20Deep-Dive.html#first-order-simplification) or [Sergey Levine's Lecture 9.1](https://www.youtube.com/watch?v=ySenCHPsKJU&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=36). To ensure the new and old policies are close, a KL constraint is included:
\[
    \text{maximise}
\quad
    \Delta(\theta, \theta_\text{old})
\quad\text{subject to}\quad
    \operatorname{KL}(\theta \mathrel{\|} \theta_\text{old})
\le \varepsilon
\]
where
\[
    \operatorname{KL}(\theta \mathrel{\|} \theta_\text{old})
:=  \mathbb E_{q \sim \mu}\bigl[
        \operatorname{KL}(\pi_\theta(\cdot \mid q) \mathrel{\|} \pi_{\theta_\text{old}}(\cdot \mid q))
    \bigr].
\]

This complication doesn't actually arise in the LLM framework: it is 'off-policy' there in the sense that there are no trajectories to depend on the current policy. Nevertheless, KL regularisation can often be helpful.

Replacing the objective and constrain with leading-order approximations, gradient ascent actually solves this exactly:
\[
    \theta_\star
=   \theta_\text{old} + \alpha F(\theta_\text{old})^{-1} \Delta'(\theta_\text{old}, \theta_\text{old})
\quad\text{where}\quad
    \alpha
=   \sqrt{2 \varepsilon / J'(\theta_\text{old})^\top F(\theta_\text{old}) J'(\theta_\text{old})},
\]
and $F(\theta) = \mathbb E_{q \sim \mu, o \sim \pi_\theta(\cdot \mid q)}[ (\nabla_\theta \log \pi_\theta(q, o)) (\nabla_\theta \log \pi_\theta(q, o))^\top ]$ is the Fisher-information matrix. This is outlined in my [RL deep-dive](RL%20Algorithms-%20Deep-Dive.html#using-gradient-ascent-and-the-fisher-information) and detailed in [Sergey Levine's Lecture 9.4](https://www.youtube.com/watch?v=QWnpF0FaKL4&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=40).



### PPO: Proximal Policy Optimisation

Calculating or estimating the Fisher matrix, or using the conjugate gradient method, is computationally expensive. PPO replaces the objective with
\[
    J_\text{PPO}(\theta; \lambda)
:=  J(\theta) - \lambda \operatorname{KL}(\theta \mathrel{\|} \theta_\text{old}),
\]
where $\lambda$ is a tunable hyperparameter. Again, the *reward* can be replaced with the *advantage*—the $\Delta$ version.

The method of Lagrange multiplies means there is *some* $\lambda$ so that the optimal $\theta$ for PPO is the same as for TRPO; finding this $\lambda$ is infeasible. In practice, *dual descent* can be used, repeating the following steps until satisfied:
\[
    \text{maximise}
\quad
    J_\text{PPO}(\theta; \lambda)
\quad\text{over}\quad
    \theta;
\qquad
    \lambda
\leftarrow
    \lambda + \alpha (\operatorname{KL}(\theta \mathrel{\|} \theta_\text{old}) - \varepsilon).
\]
Or, more informally, maximise for a given penalty $\lambda$; if the resulting KL is too large/small, then increase/decrease the penalty $\lambda$.
More details are given in [Sergey Levine's Lecture 9.3](https://www.youtube.com/watch?v=WuPauZgX7BM&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=38)

A popular extension replaces the original objective
\[
    \Delta(\theta, \theta_\text{old})
=   \mathbb E_{q \sim \mu, o \sim \pi_\theta(\cdot \mid q)}[R(q, o)]
=   \mathbb E_{q \sim \mu, o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}\biggl[
        \frac{\pi_\theta(o \mid q)}{\pi_{\theta_\text{old}}(o \mid q)}
        A^{\pi_{\theta_\text{old}}}(q, o)
    \biggr]
\]
with
\[
    \Delta_\text{PPO}^\text{clip}(\theta, \theta_\text{old}; \varepsilon)
=   \mathbb E_{q \sim \mu, o \sim \pi_{\theta_\text{old}}(\cdot \mid q)}\biggl[
        \operatorname{min–clip}_\varepsilon^\varepsilon\biggl(
            \frac{\pi_\theta(o \mid q)}{\pi_{\theta_\text{old}}(o \mid q)}
            A^{\pi_{\theta_\text{old}}}(q, o)
        \biggr)
    \biggr]
\]
where
\[
    \operatorname{min–clip}_{\varepsilon_\text{old}}^{\varepsilon_\text{high}}(x)
:=  \min\{x, \operatorname{clip}(x, 1 - \varepsilon_\text{low}, 1 + \varepsilon_\text{high})\}
\quad\text{for}\quad
    x \in \mathbb R.
\]
The clipping stops the probability ratio from changing too much, potentially over-reacting to the observed advantage. As such, the KL penalty is then sometimes removed. The default in many libraries is $\varepsilon_\text{high} = \varepsilon_\text{low} = \varepsilon = 0.2$.

Calculating the advantage, which depends on the current policy, precisely is the hard part. An estimator $\widehat A^{\pi_{\theta_\text{old}}}(q, o)$ is required. One simple option is *Monte Carlo estimation* average 100 rollouts (which themselves may be long) from the proposed state–action. This is unbiased, but often high bias and computationally heavy. To reduce bias, an estimation $V_\varphi$ the value function $V_\varphi$ is learned—typically, $\varphi$ are the weights of a neural network—and used after a few steps, such as in GAE. See my [RL deep-dive](RL%20Algorithms-%20Deep-Dive.html#estimated-advantage) or [Sergey Levine's Lecture 6.4](https://www.youtube.com/watch?v=quRjnkj-MA0&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=24) for more details.


### GRPO: Group Relative Policy Optimisation

GRPO has no critic model, using Monte Carlo style. Here, "rollouts" are simply the output (answer+reasoning) to a question.
However, instead of discarding the 100 rollouts (here, just outputs/answers to the same question) used to estimate the baseline, they are used in the update step. Concretely, the objective is
\[\begin{aligned}
&   \Delta_\text{GRPO}(\theta, \theta_\text{old}; \varepsilon, \lambda, G)
:=  \mathbb E_{q \sim \mu, o_1, ..., o_G \sim^\text{iid} \pi_{\theta_\text{old}}(\cdot \mid q)}
\\&\qquad\qquad
    \biggl[
        \frac1G \sum_{i=1}^G
        \frac1{|o_i|} \sum_{t=1}^{|o_i|}
        \operatorname{min–clip}_\varepsilon^\varepsilon\biggl(
            \frac{\pi_\theta(o_{i, t} \mid q, o_{< t})}{\pi_{\theta_\text{old}}(o_{i, t} \mid q, o_{< t})}
            \widehat A_{i,t}
        \biggr)
    -   \lambda \operatorname{KL}(\theta \mathrel{\|} \theta_\text{ref})
    \biggr],
\end{aligned}\]
where $\widehat A_{i,t}$ is an estimate of the advantage that depends only on $(q, o_1, ..., o_G)$ and $\theta_\text{ref}$ is the parameter vector of a reference model—typically, the pre-trained model, prior to RL fine-tuning. A canonical choice of estimator is
\[
    \widehat A_{i,t}
:=  \widehat A_i
:=  \frac{r_i - \operatorname{mean}(r_1, ..., r_G)}{\operatorname{std}(r_1, ..., r_G)}
\quad\text{where}\quad
    r_i = R(q, o_i).
\]
Other penalties include formatting, language and length.

The rollouts are being used both to estimate the mean reward for the question under the current policy *and* in the objective. Usually, the KL is estimated using the rollouts too: move $\operatorname{KL}(\theta \mathrel{\|} \theta_\text{ref})$ inside the average $\tfrac1G \sum_{i=1}^G (...)$ over $G$ and replace it with the (non-negative) unbiased estimator
\[
    \widehat{\operatorname{KL}}_i
:=  r^\text{ref}_i - 1 - \log r^\text{ref}_i
\quad\text{where}\quad
    r^\text{ref}_i
:=  \pi_{\theta_\text{ref}}(o_i \mid q) / \pi_\theta(o_i \mid q).
\]

A typical choice of $G$ is between $4$ and $32$. If it is too large, then any unlikely but high-reward outputs will be swamped by the typical ones. If output-generation is not the bottleneck, an option would be to sample more *to estimate the baseline* (the mean and standard deviation above) but only use a selection (perhaps ones with atypical reward?) in the update.


## Extensions to GRPO

The following algorithms are variants of and adjusments to GRPO, rather than a new approach.

### DAPO: Decoupled Clip and Dynamic Sampling Policy Optimisation

DAPO introduces five many adjustments to GRPO.

1.  **Clip Higher.**
    The upper threshold $\varepsilon_\text{high}$ in the clipping $\operatorname{min–clip}_{\varepsilon_\text{low}}^{\varepsilon_\text{high}}$ is increased. <!-- eg, [*Magistral*](http://arxiv.org/abs/2506.10910) takes $\varepsilon_\text{high} \approx 0.25$ and [*ProRL*](https://arxiv.org/abs/2505.24864) takes $\varepsilon_\text{high} = 0.4$; the lower threshold $\varepsilon_\text{low} = 0.2$ is unchanged.--> This allows low-likelihood, high-reward tokens to be boosted significantly more, increasing the exploration of the policy.

2.  **Dynamic Sampling.**
    Batch gradients are diluted by groups with (little to) no signal. DAPO filters these out, sampling more groups until the batch is complete. An alternative to oversampling is to upweight the remainder.

3.  **Token-Level Policy Gradient Loss.**
    GRPO assigns each *sample* equal weight: $G^{-1} \sum_{i=1}^G |o_i|^{-1} \sum_{t=1}^G (...)$. As such, tokens in longer reponses are underrepresented. DAPO uses equal *token* weight: $(\sum_{i=1}^G |o_i|)^{-1} \sum_{i=1}^G \sum_{t=1}^{|o_i|} (...)$.

4.  **Overlong Reward Shaping.**
    Truncating overlong responses could penalise sound reasoning. DAPO first filters overlong responses, masking their loss, then uses a *soft punishment*:
        no penalty up to a threshold,
        maximal penalty at a higher one
    and
        linear interpolation between.

5.  **Remove KL Penalty.**
    The KL penalty regulates how far the online policy can go from the reference policy. However, in practice, it diverges significantly anyway, and the KL penalty may dominate the loss. DAPO removes the KL penalty—sets $\lambda = 0$.


### ProRL: Prolonged Reinforcement Learning

ProRL utilises *clip-higher* (with $\varepsilon_\text{high} = 0.4$) and *token-level loss* from DAPO. Additionally, it (re)introduces two aspects.

1.  **KL Regularisation and Reference-Policy Reset.**
    Where DAPO removes the KL term, ProRL keeps it, but periodically hard-resets the reference weights $\theta_\text{ref}$ to a recent snapshot of the online policy and reinitialise the optimiser states. This prevents the KL penalty from dominating, whilst retaining the benefits of KL regularisation.

2.  **Prolonged RL.**
    Typically, RL implementations run for no more than a few hundred steps. ProRL uses 2000+. The hypothesis is that this gives the model time to explore and uncover new strategies.


### Magistral

Magistral uses all the extensions from DAPO (with $\varepsilon_\text{high} \approx 0.25$). Additionally, it introduces a new aspect.

1.  **Advantage Normalisation.**
    A correct answer to a hard question (which has low standard deviation) gets upweighted significantly. There is danger of noise fitting—particularly with a high $\varepsilon_\text{high}$. Instead, Magistral normalise with respect to *minibatches*. Concretely, the advantage is first *centred* wrt the question: $\widetilde A_i := r_i - \operatorname{mean}(r_1, ..., r_G)$. Then they are *normalised* wrt the minibatch: $\widehat A_i := \widetilde A_i / \operatorname{std}( (\widetilde A_i)_{i \in B})$, where $B$ is a minibatch.