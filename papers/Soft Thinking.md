---
tags:
    - 'continuous thoughts'
    - inference
    - reasoning
    - superposition
parent: 'Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space'
collections:
    - Reasoning
authors:
    - Zhang
    - He
    - Yan
    - Shen
    - Zhao
    - Wang
    - Shen
    - Wang
year: 2025
$version: 606
$libraryID: 1
$itemKey: C8WTKK7I

---
# Soft Thinking Summary

*Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space*

*   2025-05; Zhang, He, Yan, Shen, Zhao, Wang, Shen, Wang

## High-Level Ideas

*   Training-free method that enables human-like 'soft' reasoning
*   Standard Chain-of-Thought (CoT) sample one token per step and follow that path, potentially abandoning valuable alternatives
*   *Soft Thinking* generates *concept tokens* in continuous concept space corresponding to LLM-produced distribution over the vocab (at given step)
*   It's related to superposition of all possible tokens: it's a convex combination, weighted by probabilities
*   This flexibility enables the exploration of different reasoning paths, and avoids making hard (consequential) decisions too early
*   A *cold stop* (entropy threshold) is used to mitigate out-of-distribution issues

## Related Papers

*   *Text Generation Beyond Discrete Token Sampling* ("Mixture of Inputs")

    *   This paper seems to suggest that setting the weights to the probabilities, as in *Soft Thinking* and referred to as *Direct Mixture* here, leads to a degradation of performance, but they don’t (appear to) employ the cold stop; details are in Section 6.1 of that paper

*   *Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought*

## Some Details

Instead of injecting the random, one-hot encoded token, after embedding, into the LLM, the concept token is injected back:

> inject $\sum_{k=1}^{|V|} \mathsf{ct}[k] e(k)$

where $\mathsf{ct}[k]$/$e(k)$ is the selection probability/embedding of the $k$-th vocab item.

The theoretical justification is based on linear approximations in the expression

> $p(y \mid x) = \sum_{t_1} p(t_1 \mid x) \sum_{t_2} p(t_2 \mid x, t_1) ... \sum_{t_m} p(t_m \mid x, t_{1:m-1}) p(y \mid x, t_{1:m})$;

here, $t_j$ is the $j$-th (intermediate reasoning) token. To emphasise, $p(\cdot \mid x, t_{1:j})$ is the probability of the next token given the input $x$ and intermediate reasoning tokens $t_1, ..., t_j$. Once some stopping criterion is achieved, the model outputs an *answer*, denoted $y$.

This expansion entails exponentially-in-$m$ many paths, indexed by the choice of intermediate reasoning tokens. If we only expanded one layer,

> $p(y \mid x) = \sum_{t_1} p(t_1 \mid x) p(y \mid x, t_1)$.

In expectation,

> $\mathsf{ct}_1 = \mathbb E[t_1] = \sum_{t_1} p(t_1 \mid x) t_1 = p(\cdot \mid x)$.

Linearising the previous expression about its mean, (ie, replacing random $t_1$ by its non-random mean $\mathsf{ct}_1$),

> $p(y \mid x) = \sum_{t_1} p(t_1 \mid x) p(y \mid x, t_1) \approx p(y \mid x, \sum_{t_1} p(t_1 \mid x) t_1) = p(y \mid x, \mathsf{ct}_1)$.

The approximation is repeated given $\mathsf{ct}_1$:

> $p(y \mid x, \mathsf{ct}_1) = \sum_{t_2} p(t_2 \mid \mathsf{ct}_1) p(y \mid x, \mathsf{ct}_1, \mathsf{ct}_2) \approx p(y \mid x, \mathsf{ct}_1, \mathsf{ct}_2)$.

Iterating,

> $p(y \mid x) \approx p(y \mid x, \mathsf{ct}_1, \mathsf{ct}_2, ..., \mathsf{ct}_m)$.

In contrast, discrete CoT replaces each summation $\sum_{t_j} p(t_j \mid \cdot)$ with sampling a single token, which discards the mass from all other paths. *Soft thinking* preserves the distribution through the concept tokens, whilst collapsing the exponential path summation to a single forward pass.

A *cold stop* is also implemented, where the entropy is tracked in real time. This is to address issues in which the continuous concept tokens place the model in an out-of-distribution regime. If the entropy (ie, uncertainty) of a concept token drops below a threshold, the process stops. A basic ablation study is conducted; with details below.

## Results

In summary, I'd suggest that the accuracy increase is modest, but the generation length reduction is significant.

The first table is the *accuracy* (higher → better) and the second the *generation length* (lower → better).

![\<img alt="Table of results" data-attachment-key="TXT5PX7V" width="859" height="731" src="attachments/TXT5PX7V.png" ztype="zimage"> | 859](attachments/TXT5PX7V.png){ style="display: block; margin: 0 auto" }

<!---
| Model                             | Method        | MATH 500 | AIME 2024 | GSM8K | GPQA-Diamond | Average |
| --------------------------------- | ------------- | -------- | --------- | ----- | ------------ | ------- |
| **QwQ-32B**                       | CoT baseline  | 97.66    | 76.88     | 96.67 | 64.17        | 83.84   |
|                                   | CoT greedy    | 97.00    | 80.00     | 96.57 | 65.17        | 84.68   |
|                                   | Soft Thinking | 98.00    | 83.33     | 96.81 | 67.17        | 86.32   |
| **DeepSeek-R1-Distill-Qwen-32B**  | ...           |          |           |       |              |         |
| ...                               |               |          |           |       |              |         |
| **DeepSeek-R1-Distill-Llama-70B** | ...           |          |           |       |              |         |
| ...                               |               |          |           |       |              |         |
--->

The *soft thinking* results all utilise a *cold stop*, with threshold $\tau$ optimised for the problem at hand. An ablation study is conducted regarding *cold stop*—namely, $\tau = 0$ is forced, ensuring the cold stop is never activated.

The four lines in the table below correspond to different strategies.

*   COCONUT-TF: the training-free COCONUT approach simply feeds the previous hidden state as the next input embedding.

*   Average Embedding: the (embeddings of the) top 5 tokens are averaged and fed.

*   w/o Cold Stop: *soft thinking* without cold stop—ie,  $\tau = 0$  enforced.

*   w/ Cold Stop: full *soft thinking*, with cold-stop threshold  $\tau$  optimised (swept).

![\<img alt="Ablation study" data-attachment-key="IVA9NW7K" width="1123" height="260" src="attachments/IVA9NW7K.png" ztype="zimage"> | 1123](attachments/IVA9NW7K.png)

## Limitations and Concerns

...
