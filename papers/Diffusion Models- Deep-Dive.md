---
tags:
    - diffusion
    - RL
    - training
    - own
method: diffusion
title: 'Diffusion Models: Deep Dive'
lab: OT
date: 202603
---

# Diffusion Models: A Precise Account

# Part I: Probabilistic set-up

## I.1. Notation

All random variables live in $\mathbb{R}^d$. We write $X_0, X_1, \ldots, X_T$ for the random variables in the forward chain, and use lowercase $x, y$, etc. as dummy arguments of densities. We write $\mathcal{N}(m, \Sigma)$ for the Gaussian distribution with mean $m$ and covariance $\Sigma$, and $\mathcal{N}(x; m, \Sigma)$ for its density evaluated at $x$. Densities carry explicit subscripts to avoid ambiguity:

- $q_t(x)$: marginal density of $X_t$
- $q_{t|s}(x \mid y)$: conditional density of $X_t$ given $X_s = y$
- $q_{t|s,r}(x \mid y, z)$: conditional density of $X_t$ given $X_s = y$ and $X_r = z$

The learned reverse process defines a separate chain $Y_T, Y_{T-1}, \ldots, Y_0$ with densities $p_\theta$, using analogous subscript conventions.

## I.2. Forward process

Let $q_0 = p_{\mathrm{data}}$ be the data distribution. Fix a schedule $0 < \beta_1, \ldots, \beta_T < 1$. Define the forward chain by $X_0 \sim q_0$ and

$$q_{t|t-1}(x \mid y) = \mathcal{N}(x;\; \sqrt{1-\beta_t}\,y,\; \beta_t\,I), \qquad t = 1, \ldots, T.$$

Set $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. By induction (composing affine Gaussian maps), the marginal conditional on $X_0$ has the closed form

$$q_{t|0}(x \mid x_0) = \mathcal{N}(x;\; \sqrt{\bar{\alpha}_t}\,x_0,\; (1 - \bar{\alpha}_t)\,I).$$

Equivalently, $X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$ is independent of $X_0$. The schedule is chosen so that $\bar{\alpha}_T \approx 0$, ensuring $q_T \approx \mathcal{N}(0, I)$.

The (unconditional) marginal density of $X_t$ is

$$q_t(x) = \int_{\mathbb{R}^d} q_{t|0}(x \mid x_0)\,q_0(x_0)\,\mathrm{d}x_0.$$

## I.3. The true reverse transitions

By Bayes' rule on the forward chain, for $t \geq 2$:

$$q_{t-1|t}(x \mid y) = \frac{q_{t|t-1}(y \mid x)\,q_{t-1}(x)}{q_t(y)}.$$

This involves the intractable marginals $q_{t-1}$ and $q_t$, so we cannot compute $q_{t-1|t}$ directly. However, if we additionally condition on $X_0$, everything becomes Gaussian. Applying Bayes' rule within the forward chain conditioned on $X_0 = x_0$:

$$q_{t-1|t,0}(x \mid y, x_0) = \frac{q_{t|t-1}(y \mid x)\,q_{t-1|0}(x \mid x_0)}{q_{t|0}(y \mid x_0)}.$$

All three factors on the right are Gaussian, so the left-hand side is Gaussian. Completing the square gives

$$q_{t-1|t,0}(x \mid y, x_0) = \mathcal{N}(x;\; \tilde{\mu}_t(y, x_0),\; \tilde{\beta}_t\,I)$$

where

$$\tilde{\mu}_t(y, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\,y, \qquad \tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\,\beta_t.$$

The true (unconditioned-on-$X_0$) reverse kernel is then the mixture

$$q_{t-1|t}(x \mid y) = \int_{\mathbb{R}^d} q_{t-1|t,0}(x \mid y, x_0)\,q_{0|t}(x_0 \mid y)\,\mathrm{d}x_0$$

where $q_{0|t}(x_0 \mid y) \propto q_{t|0}(y \mid x_0)\,q_0(x_0)$ is the posterior over clean images given a noisy observation — a mixture over the entire data manifold and completely intractable. The tractable Gaussian $q_{t-1|t,0}$, however, will be central to both training (Part II) and sampling (Part III): it is the target each learned reverse step tries to approximate, and the bridge posterior that determines the sampling geometry.

# Part II: Training

## II.1. The learned reverse process and the ELBO

We define a reverse chain $Y_T, \ldots, Y_0$ with $Y_T \sim \mathcal{N}(0, I)$ and learnable transitions

$$p_{\theta,t}(x \mid y) = \mathcal{N}(x;\; \mu_\theta(y, t),\; \sigma_t^2\,I), \qquad t = T, T-1, \ldots, 1$$

where $\sigma_t^2$ is fixed (eg to $\beta_t$ or $\tilde{\beta}_t$). The marginal density of $Y_0$ under this chain is

$$p_\theta(x) = \int_{\mathbb{R}^{dT}} p(y_T)\prod_{t=1}^{T} p_{\theta,t}(y_{t-1} \mid y_t)\,\mathrm{d}y_{1:T}$$

where $p$ denotes the density of $\mathcal{N}(0,I)$ and we have written $y_0 = x$. We want to maximise $\log p_\theta(x_0)$ for data points $x_0 \sim q_0$.

### Derivation of the ELBO

The evidence lower bound (ELBO) is derived as follows.

**Step 1.** Write the marginal as an expectation under the forward posterior:

$$\log p_\theta(x_0) = \log \int \frac{p(x_T)\prod_{t=1}^T p_{\theta,t}(x_{t-1}\mid x_t)}{q_{1:T|0}(x_{1:T}\mid x_0)}\;q_{1:T|0}(x_{1:T}\mid x_0)\,\mathrm{d}x_{1:T}$$

where $q_{1:T|0}(x_{1:T}\mid x_0) = \prod_{t=1}^T q_{t|t-1}(x_t \mid x_{t-1})$ is the joint forward density.

**Step 2.** Apply Jensen's inequality ($\log$ is concave, $q_{1:T|0}(\cdot \mid x_0)$ is a probability measure):

$$\log p_\theta(x_0) \geq \mathbb{E}_{q_{1:T|0}(\cdot \mid x_0)}\!\left[\log\frac{p(X_T)\prod_{t=1}^T p_{\theta,t}(X_{t-1}\mid X_t)}{\prod_{t=1}^T q_{t|t-1}(X_t \mid X_{t-1})}\right] =: -\mathcal{L}(x_0;\theta).$$

So $\mathcal{L}(x_0;\theta)$ is an upper bound on $-\log p_\theta(x_0)$, or equivalently $-\mathcal{L}$ is a lower bound on $\log p_\theta(x_0)$.

**Step 3.** Decompose $\mathcal{L}$. Inside the expectation, write

$$\log \frac{\prod_{t=1}^T q_{t|t-1}(X_t \mid X_{t-1})}{p(X_T)\prod_{t=1}^T p_{\theta,t}(X_{t-1} \mid X_t)} = -\log p(X_T) + \sum_{t=1}^T \log\frac{q_{t|t-1}(X_t \mid X_{t-1})}{p_{\theta,t}(X_{t-1} \mid X_t)}.$$

For $t \geq 2$, use Bayes' rule within the forward chain conditioned on $X_0$:

$$q_{t|t-1}(X_t \mid X_{t-1}) = \frac{q_{t-1|t,0}(X_{t-1} \mid X_t, X_0)\;q_{t|0}(X_t \mid X_0)}{q_{t-1|0}(X_{t-1} \mid X_0)}$$

to rewrite each term ($t \geq 2$) as

$$\begin{aligned}
\log\frac{q_{t|t-1}(X_t \mid X_{t-1})}{p_{\theta,t}(X_{t-1}\mid X_t)}
&= \log\frac{q_{t-1|t,0}(X_{t-1}\mid X_t, X_0)}{p_{\theta,t}(X_{t-1}\mid X_t)} \\
&\quad + \log q_{t|0}(X_t\mid X_0) - \log q_{t-1|0}(X_{t-1}\mid X_0).
\end{aligned}$$

Summing from $t = 2$ to $T$, the $\log q_{t|0}$ terms telescope:

$$\sum_{t=2}^T \bigl[\log q_{t|0}(X_t \mid X_0) - \log q_{t-1|0}(X_{t-1}\mid X_0)\bigr] = \log q_{T|0}(X_T\mid X_0) - \log q_{1|0}(X_1\mid X_0).$$

The $t=1$ term contributes $\log q_{1|0}(X_1\mid X_0) - \log p_{\theta,1}(X_0 \mid X_1)$. Collecting everything:

$$\begin{aligned}
\mathcal{L}(x_0;\theta) = \mathbb{E}\!\Bigl[
&\underbrace{D_{\mathrm{KL}}\!\bigl(q_{T|0}(\cdot\mid X_0)\;\|\;p\bigr)}_{L_T} - \underbrace{\log p_{\theta,1}(X_0\mid X_1)}_{-L_0} \\[4pt]
&+ \sum_{t=2}^{T}\underbrace{D_{\mathrm{KL}}\!\bigl(q_{t-1|t,0}(\cdot\mid X_t, X_0)\;\|\;p_{\theta,t}(\cdot\mid X_t)\bigr)}_{L_{t-1}}\;\Bigr]
\end{aligned}$$

where the outer expectation is over $X_0 \sim q_0$ and $X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$ with $\varepsilon\sim\mathcal{N}(0, I)$. Note: $L_T$ has no learnable parameters; $L_0$ is a reconstruction term; the interesting terms are $L_1, \ldots, L_{T-1}$.

## II.2. Reducing $L_{t-1}$ to noise prediction

Each $L_{t-1}$ (for $t \geq 2$) is a KL divergence (Kullback–Leibler divergence) between two Gaussians with the same covariance when $\sigma_t^2 = \tilde{\beta}_t$, reducing to

$$L_{t-1} = \frac{1}{2\tilde{\beta}_t}\bigl\|\tilde{\mu}_t(X_t, X_0) - \mu_\theta(X_t, t)\bigr\|^2 + \text{const}.$$

Substitute $X_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\bigl(X_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon\bigr)$ into the expression for $\tilde{\mu}_t$:

$$\tilde{\mu}_t(X_t, \varepsilon) = \frac{1}{\sqrt{\alpha_t}}\!\left(X_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon\right).$$

Parameterise the model mean as

$$\mu_\theta(y, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(y - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(y, t)\right)$$

so that $L_{t-1} \propto \|\varepsilon - \varepsilon_\theta(X_t, t)\|^2$. The DDPM (denoising diffusion probabilistic model) simplified objective, which drops the $t$-dependent prefactor and sums uniformly over $t$, is

$$L_{\mathrm{simple}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}\{1,\ldots,T\},\;X_0 \sim q_0,\;\varepsilon\sim\mathcal{N}(0, I)}\!\bigl[\|\varepsilon - \varepsilon_\theta(X_t, t)\|^2\bigr]$$

where $X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$.

## II.3. Conditional training and caption dropout

Given image-caption pairs $(X_0, C) \sim p_{\mathrm{data}}$, the forward process acts only on the image: $q_{t|0}(x \mid x_0)$ is unchanged, and the caption $C$ is carried along as an unperturbed label. The noise predictor becomes $\varepsilon_\theta(y, t, c)$ and the training loss is

$$L_{\mathrm{simple}}(\theta) = \mathbb{E}_{t,\,(X_0, C),\,\varepsilon}\!\bigl[\|\varepsilon - \varepsilon_\theta(X_t, t, C)\|^2\bigr].$$

This is the only change: condition the network on $c$ and train with the same squared-error objective.

To enable classifier-free guidance (CFG) at inference (Section III.2), one additionally trains the network to operate without a caption. During training, the caption $C$ is replaced with a null token $\varnothing$ independently with probability $p_{\mathrm{uncond}} \approx 0.1$. Writing $\tilde{C}$ for the resulting input ($C$ with probability $1 - p_{\mathrm{uncond}}$, $\varnothing$ otherwise), the loss remains $\mathbb{E}[\|\varepsilon - \varepsilon_\theta(X_t, t, \tilde{C})\|^2]$. After convergence, the single network approximates two score functions depending on its third argument:

$$\varepsilon_\theta(y, t, c) \approx -\sqrt{1-\bar{\alpha}_t}\,\nabla_y \log q_t(y \mid c), \qquad \varepsilon_\theta(y, t, \varnothing) \approx -\sqrt{1-\bar{\alpha}_t}\,\nabla_y \log q_t(y).$$

The first is the score of the conditional marginal $q_t(\cdot \mid c)$; the second is the score of the unconditional marginal $q_t$. How these are combined at inference time is described in Section III.2.

## II.4. What training actually does, and what it learns

### Training is just denoising

The operational content of training is: sample $X_0 \sim q_0$, sample $t \sim \mathrm{Unif}\{1,\ldots,T\}$ and $\varepsilon \sim \mathcal{N}(0, I)$, form $X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$, and regress $\varepsilon_\theta(X_t, t)$ against $\varepsilon$. No $X_{t-1}$ is needed; the closed-form marginal $q_{t|0}$ lets you jump directly to any noise level without simulating the chain. Note that predicting $\varepsilon$ and predicting $X_0$ are equivalent given $(X_t, t)$, since $\hat{X}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(X_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta)$.

### Why denoising gives a generative model

A standalone denoiser applied once to $X_T \sim \mathcal{N}(0, I)$ would produce the minimum mean squared error (MMSE) estimate $\mathbb{E}[X_0 \mid X_T]$ — a blurry average, not a sharp sample. The ELBO (Section II.1) ensures that training a denoiser at every noise level does more: each $L_{t-1}$ fits $p_{\theta,t}(\cdot \mid y)$ to the forward posterior $q_{t-1|t,0}(\cdot \mid y, X_0)$ averaged over $X_0$, calibrating a chain whose $T$ transitions from pure noise produce samples from $\approx q_0$. No individual step needs to do anything dramatic — at large $t$, $\hat{x}_0$ is poor but $\tilde{\mu}_t$ barely trusts it; at small $t$, the estimate is accurate and the step commits.

### The model learns the marginal reverse, not the trajectory

For each $t$, the loss $L_{t-1}$ asks the single Gaussian $p_{\theta,t}(\cdot \mid y)$ to match $q_{t-1|t,0}(\cdot \mid y, X_0)$, but the outer expectation averages over $X_0 \sim q_{0|t}(\cdot \mid y)$. The effective target for $\mu_\theta(y, t)$ is therefore

$$\begin{aligned}
\operatorname*{arg\,min}_{\mu} \;\mathbb{E}_{X_0 \sim q_{0|t}(\cdot|y)}\!\bigl[\|\tilde{\mu}_t(y, X_0) - \mu\|^2\bigr]
&= \mathbb{E}_{X_0 \sim q_{0|t}(\cdot|y)}\!\bigl[\tilde{\mu}_t(y, X_0)\bigr] \\
&= \tilde{\mu}_t\!\bigl(y,\;\mathbb{E}[X_0 \mid X_t = y]\bigr)
\end{aligned}$$

where the last equality uses linearity of $\tilde{\mu}_t$ in $x_0$. In the noise parameterisation, this is equivalent to $\varepsilon_\theta(y, t) \to \mathbb{E}[\varepsilon \mid X_t = y]$.

The model learns the MMSE denoiser: the conditional expectation of the noise (or equivalently of $X_0$) given $X_t = y$. The single Gaussian $p_{\theta,t}(\cdot \mid y)$ approximates the intractable mixture $q_{t-1|t}(\cdot \mid y)$ by matching its mean.

The ELBO ties every learned transition to the data distribution $q_0$: the model can only generate images that resemble training data. Part IV replaces this objective with an arbitrary reward.

# Part III: Sampling

## III.1. The sampling algorithm and its structure

Given a trained noise predictor $\varepsilon_\theta$, Algorithm 2 of Ho et al. (2020) generates images by initialising $x_T \sim \mathcal{N}(0, I)$ and iterating for $t = T, T-1, \ldots, 1$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right) + \sigma_t\,z, \qquad z \sim \mathcal{N}(0, I)$$

(with $z = 0$ at $t=1$). This decomposes as $x_{t-1} = m_t(x_t) + \sigma_t\,z$: a deterministic target $m_t(x_t)$ plus isotropic Gaussian noise. Each step is therefore a draw from $\mathcal{N}(m_t(x_t),\,\sigma_t^2 I)$ — an isotropic Gaussian with exact, closed-form log-density. This structure is what makes the policy gradient computation in Part IV tractable.

Sections III.3 and III.4 derive $m_t$ and $\sigma_t$. First, Section III.2 describes how conditioning on a caption enters at inference time via classifier-free guidance.

## III.2. Classifier-free guidance

Sampling with the conditional model $\varepsilon_\theta(x_t, t, c)$ alone would approximate $q_0(\cdot \mid c)$. Classifier-free guidance (CFG), introduced by Ho and Salimans (2022), sharpens this by combining the conditional and unconditional score estimates from Section II.3.

At inference, one forms the guided noise estimate

$$\hat{\varepsilon}(y, t, c) = (1+w)\,\varepsilon_\theta(y, t, c) - w\,\varepsilon_\theta(y, t, \varnothing)$$

for a guidance weight $w > 0$, and uses $\hat{\varepsilon}$ in place of $\varepsilon_\theta$ in the sampling step. Expressing this in terms of scores (up to the factor $-\sqrt{1-\bar\alpha_t}$), the guided estimate follows

$$(1+w)\,\nabla_y \log q_t(y \mid c) - w\,\nabla_y \log q_t(y).$$

Applying Bayes' rule — $\nabla_y \log q_t(y \mid c) = \nabla_y \log q_t(c \mid y) + \nabla_y \log q_t(y)$, since $\nabla_y \log q_t(c) = 0$ — this becomes

$$\nabla_y\!\bigl[\log q_t(y) + (1+w)\log q_t(c \mid y)\bigr].$$

The unconditional score $\nabla_y \log q_t(y)$ keeps the sample on the image manifold; the term $(1+w)\log q_t(c \mid y)$ acts as an amplified implicit classifier steering toward images strongly associated with caption $c$. At $w = 0$ this reduces to standard conditional sampling; increasing $w$ concentrates the effective distribution on high-$q_t(c \mid y)$ modes, improving caption fidelity at the cost of diversity. The guided score field does not, in general, integrate to a normalised density — CFG is a heuristic that manipulates the score at inference time, with no formal objective being optimised. Part IV takes a different approach, defining an explicit reward and optimising it via RL.

## III.3. The deterministic target: where the step lands

The target $m_t(x_t)$ is a weighted average of the current noisy state $x_t$ and an estimate of the clean image $x_0$. Define the MMSE clean estimate from the noise predictor:

$$\hat{x}_0(x_t) = \frac{1}{\sqrt{\bar{\alpha}_t}}\!\bigl(x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta(x_t, t)\bigr).$$

Substituting $\hat{x}_0$ into $\tilde{\mu}_t(x_t, \hat{x}_0)$ — the forward posterior mean from Section I.3, with $\hat{x}_0$ in place of the true clean image:

$$\tilde{\mu}_t(x_t, \hat{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\hat{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t.$$

Using $\frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} = \frac{1}{\sqrt{\alpha_t}}$, the coefficient of $x_t$ becomes $\frac{\beta_t + \alpha_t(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}$. The numerator simplifies: $\beta_t + \alpha_t - \alpha_t\bar{\alpha}_{t-1} = (1-\alpha_t) + \alpha_t - \bar{\alpha}_t = 1 - \bar{\alpha}_t$, so the coefficient is $\frac{1}{\sqrt{\alpha_t}}$. The coefficient of $\varepsilon_\theta$ is $\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}$. Therefore:

$$\tilde{\mu}_t(x_t, \hat{x}_0) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right)$$

which is exactly $m_t(x_t)$. The sampling step is therefore

$$x_{t-1} = \tilde{\mu}_t(x_t, \hat{x}_0(x_t)) + \sigma_t\,z$$

ie, it draws from the forward posterior $q_{t-1|t,0}(\cdot \mid x_t, \hat{x}_0)$, treating the MMSE estimate as the true clean image.

### Contrast with "estimate $x_0$ then re-corrupt"

If sampling instead jumped to $\hat{x}_0$ and re-injected noise to reach level $t{-}1$, the step would be $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0(x_t) + \sqrt{1-\bar{\alpha}_{t-1}}\,z$. This forgets $x_t$ entirely, independently re-adding all noise for level $t{-}1$ from scratch.

The actual step is far more conservative. In $\tilde{\mu}_t$, the weight on $\hat{x}_0$ is $\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}$, which is small when $\bar{\alpha}_{t-1} \approx 0$ (large $t$, most noise remaining) and approaches $1$ as $\bar{\alpha}_{t-1} \to 1$ (small $t$, little noise remaining). Early steps barely trust $\hat{x}_0$; late steps commit to it. This conservatism is essential: $\hat{x}_0$ is poor at large $t$, but the step does not rely on it.

## III.4. The bridge interpretation

The weights in $\tilde{\mu}_t$ and the variance $\tilde{\beta}_t$ arise from the bridge structure of the forward chain.

The forward chain $(X_0, X_1, \ldots, X_T)$ conditioned on $X_0 = x_0$ is a Gaussian Markov chain (OU-like, not a random walk, due to the $\sqrt{1-\beta_t}$ contraction). Pinning both endpoints — conditioning on $X_0 = x_0$ and $X_t = x_t$ — gives a jointly Gaussian distribution over the intermediate variables $(X_1, \ldots, X_{t-1})$, which is itself Markov with fixed endpoints: a Gaussian Markov bridge. The density $q_{t-1|t,0}(\cdot \mid x_t, x_0)$ is the one-step-back marginal of this bridge, with mean $\tilde{\mu}_t(x_t, x_0)$ and variance $\tilde{\beta}_t\,I$, both determined by the geometry of the forward chain.

With $\sigma_t^2 = \tilde{\beta}_t$, the sampling step is precisely

$$x_{t-1} \sim q_{t-1|t,0}(\;\cdot\;\mid\;x_t,\;\hat{x}_0(x_t)),$$

ie, a draw from the bridge posterior, treating the MMSE estimate as the true clean image. In practice one never constructs the bridge as an object — just evaluate $\tilde{\mu}_t(x_t, \hat{x}_0)$ and add noise $\sqrt{\tilde{\beta}_t}\,z$ — but the bridge interpretation explains why these particular weights and this particular variance are correct.

# Part IV: Finetuning with reinforcement learning

Parts II and III optimise a single objective: matching the data distribution $q_0$ via the ELBO. But many applications care about a downstream property of the generated image — aesthetic quality, compressibility, prompt-image alignment — that is not captured by data likelihood. This section recasts the sampling chain from Part III as a Markov decision process (MDP), enabling policy gradient methods to optimise an arbitrary reward $r(x_0, c)$ directly. The framework is due to Black et al. (2024), who call the resulting algorithm denoising diffusion policy optimisation (DDPO).

## IV.1. The reward maximisation objective

Given a distribution over captions $\mathcal{C}$ and a reward function $r: \mathbb{R}^d \times \mathcal{C} \to \mathbb{R}$, we want to solve

$$\max_\theta \; J(\theta) = \mathbb{E}_{c \sim \mathcal{C}}\;\mathbb{E}_{x_0 \sim p_\theta(\cdot \mid c)}\!\bigl[r(x_0, c)\bigr]$$

where $p_\theta(x_0 \mid c) = \int p(x_T)\prod_{t=1}^{T} \pi_{\theta,t}(x_{t-1} \mid x_t, c)\;\mathrm{d}x_{1:T}$ is the marginal over final images induced by the sampling chain. Since this integral is over $\mathbb{R}^{dT}$, neither $p_\theta(x_0 \mid c)$ nor its gradient with respect to $\theta$ can be evaluated.

A naive approach (reward-weighted regression, or RWR) applies REINFORCE directly to this marginal:

$$\nabla_\theta J = \mathbb{E}_{c \sim \mathcal{C},\; x_0 \sim p_\theta(\cdot \mid c)}\!\bigl[r(x_0, c)\;\nabla_\theta \log p_\theta(x_0 \mid c)\bigr].$$

This is exact in principle, but $\log p_\theta(x_0 \mid c)$ is the intractable log-marginal. In practice, RWR substitutes the ELBO (or another bound) for $\log p_\theta(x_0 \mid c)$, which introduces bias: the gradient is no longer that of $J$ but of a surrogate objective. DDPO eliminates this approximation.

## IV.2. The multi-step MDP

Recall from Part III that each sampling step is a draw from an isotropic Gaussian:

$$\pi_{\theta,t}(x_{t-1} \mid x_t, c) = \mathcal{N}\!\bigl(x_{t-1};\; \tilde{\mu}_t(x_t, \hat{x}_0(x_t, c)),\; \sigma_t^2\,I\bigr)$$

where $\hat{x}_0(x_t, c) = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta(x_t, t, c))$ and $\tilde{\mu}_t$ is the bridge target from Section III.3. (The notation $\pi_{\theta,t}$ replaces $p_{\theta,t}$ from Part II to match MDP conventions; the object is the same.) DDPO maps the sampling chain to the following MDP:

- **State**: $(x_t, t, c)$. Initial state: $x_T \sim \mathcal{N}(0, I)$, $c \sim \mathcal{C}$.
- **Action**: $x_{t-1} \in \mathbb{R}^d$.
- **Policy**: $\pi_{\theta,t}(x_{t-1} \mid x_t, c)$ as above.
- **Transition**: deterministic — the next state is $(x_{t-1}, t-1, c)$. All stochasticity is in the policy.
- **Reward**: $r(x_0, c)$ at the terminal step; zero otherwise.

A trajectory is $\tau = (x_T, x_{T-1}, \ldots, x_0)$ — one run of the sampling chain.

## IV.3. Factorised likelihoods and the policy gradient

The trajectory log-probability factorises as

$$\log p_\theta(\tau \mid c) = \log p(x_T) + \sum_{t=1}^{T} \log \pi_{\theta,t}(x_{t-1} \mid x_t, c)$$

where each term is an exact isotropic Gaussian log-density:

$$\log \pi_{\theta,t}(x_{t-1} \mid x_t, c) = -\frac{d}{2}\log(2\pi\sigma_t^2) - \frac{1}{2\sigma_t^2}\bigl\|x_{t-1} - \tilde{\mu}_t(x_t, \hat{x}_0(x_t, c))\bigr\|^2.$$

This is the central advantage: the intractable $\nabla_\theta \log p_\theta(x_0 \mid c)$ from Section IV.1 has been replaced by a sum of $T$ exact, closed-form terms. Applying REINFORCE to the trajectory gives

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\cdot \mid c)}\!\left[r(x_0, c)\sum_{t=1}^{T}\nabla_\theta \log \pi_{\theta,t}(x_{t-1} \mid x_t, c)\right]$$

where each $\nabla_\theta \log \pi_{\theta,t}$ propagates through $\tilde{\mu}_t$ and hence through $\varepsilon_\theta$:

$$\nabla_\theta \log \pi_{\theta,t}(x_{t-1} \mid x_t, c) = \frac{1}{\sigma_t^2}\bigl(x_{t-1} - \tilde{\mu}_t(x_t, \hat{x}_0(x_t, c))\bigr)\;\nabla_\theta \tilde{\mu}_t(x_t, \hat{x}_0(x_t, c)).$$

In practice, one samples trajectories from $p_\theta$, evaluates $r(x_0, c)$, and performs a gradient step.

## IV.4. On-policy and off-policy variants

The gradient in Section IV.3 requires trajectories sampled from the current policy $p_\theta$. DDPO-SF (score function) implements this directly: sample a batch of trajectories, compute the REINFORCE gradient, update $\theta$, discard the batch. Each trajectory is used for exactly one gradient step.

DDPO-IS (importance sampling) reuses trajectories across multiple updates. Given trajectories sampled from a previous policy $p_{\theta_{\mathrm{old}}}$, the gradient under the current $p_\theta$ is reweighted by the trajectory importance ratio

$$\frac{p_\theta(\tau \mid c)}{p_{\theta_{\mathrm{old}}}(\tau \mid c)} = \prod_{t=1}^{T} \frac{\pi_{\theta,t}(x_{t-1} \mid x_t, c)}{\pi_{\theta_{\mathrm{old}},t}(x_{t-1} \mid x_t, c)}$$

which factorises into per-step Gaussian density ratios, each exact and closed-form. As $\theta$ drifts from $\theta_{\mathrm{old}}$, the importance weights become high-variance. Following PPO, DDPO-IS clips the per-step ratios to $[1-\epsilon, 1+\epsilon]$, preventing large updates from stale trajectories. This improves sample efficiency at the cost of bias from clipping.

## IV.5. Credit assignment

The reward is sparse: $r(x_0, c)$ arrives only at the terminal step, but the policy makes $T$ decisions. Two features make credit assignment tractable. First, per-prompt baselines: for each caption $c$, the mean reward across sampled trajectories is subtracted, so the gradient reinforces trajectories that outperform the per-prompt average. Second, near-determinism: since $\sigma_t$ is small, the trajectory barely branches, so the terminal reward is a smooth function of early decisions. In the limit $\sigma_t \to 0$ (DDIM), the trajectory becomes fully deterministic — eliminating credit assignment but also the stochasticity REINFORCE requires.

## IV.6. Reward hacking and KL regularisation

The objective $J(\theta)$ contains no data-matching term: if $r$ is an imperfect proxy for the true goal, unconstrained optimisation will exploit the gap, typically degrading image quality to game the metric (eg, compressibility rewards produce featureless blobs that achieve optimal file size but contain no meaningful content).

The principled fix adds a KL penalty against the pretrained reference $p_{\mathrm{ref}}$:

$$\max_\theta \; \mathbb{E}_{c,\; x_0 \sim p_\theta(\cdot|c)}\!\bigl[r(x_0, c)\bigr] - \beta\, D_{\mathrm{KL}}\!\bigl(p_\theta(\cdot | c) \,\|\, p_{\mathrm{ref}}(\cdot | c)\bigr).$$

This is more than a constraint on how far $p_\theta$ can drift. The KL-regularised objective has a closed-form optimum:

$$p^*(x_0 \mid c) \propto p_{\mathrm{ref}}(x_0 \mid c)\,\exp\!\bigl(r(x_0, c)/\beta\bigr).$$

This Gibbs reweighting upweights images that are both plausible under $p_{\mathrm{ref}}$ and high-reward, while suppressing high-reward but implausible images. The parameter $\beta$ controls the trade-off: large $\beta$ keeps $p^*$ close to $p_{\mathrm{ref}}$; small $\beta$ concentrates on reward-maximising modes that still have reference support. The KL term actively shapes the target distribution — defining what the optimal policy is, not merely limiting how far optimisation can go. If $r$ is sufficiently misspecified, however, even $p^*$ will exploit the gap within the support of $p_{\mathrm{ref}}$.
