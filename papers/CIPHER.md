---
tags:
    - 'continuous thoughts'
    - inference
    - reasoning
    - superposition
parent: 'Let Models Speak Ciphers: Multiagent Debate through Embeddings'
authors:
    - Pham
    - Liu
    - Yang
    - Chen
    - Liu
    - Yuan
    - Plummer
    - Wang
    - Yang
year: 2023

---
# CIPHER Summary

[*Let Models Speak Ciphers: Multiagent Debate through Embeddings*](http://arxiv.org/abs/2310.06272)

-   2023-10; Pham, Liu, Yang, Chen, Liu, Yuan, Plummer, Wang, Yang

## High-Level Ideas

-   Training-free method that enables human-like 'soft' reasoning
-   Standard Chain-of-Thought (CoT) sample one token per step and follow that path, potentially abandoning valuable alternatives
-   CIPHER feeds the weighted average of all embedded tokens, weighted by the (softmax) LLM probability distribution, back in
-   It also uses a debate approach:
    -   multiple agents provide a CIPHER response
    -   all of these are concatenated with the current prompt
    -   this is repeated for some number of rounds
-   Bayesian optimisation is utilised to select the temperature

## Related Papers

-   [*Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space*](http://arxiv.org/abs/2505.15778)
    -   *soft thinking* seems to be *exactly* the same idea
    -   current paper also has a debate element

-   [*Text Generation Beyond Discrete Token Sampling*](https://arxiv.org/abs/2505.14827) ("Mixture of Inputs")

-   [*Training Large Language Models to Reason in a Continuous Latent Space*](https://arxiv.org/abs/2412.06769) (COCONUT)
    -   feeds last hidden layer straight back into input
    -   doesn't go via (a weighted average in) the vocab space

-   [*Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought*](http://arxiv.org/abs/2505.12514)
    -   theoretical analysis of COCONUT, focused on graph-reachability
    -   shows COCONUT is able to keep possible solutions in superposition


## Some Details

There are two main aspects of this paper:

-   a new, 'non-verbal' approach CoT, feeding a mixture of tokens back in;
-   a debate mechanism for improving responses.


### Mixture of Tokens

A standard causal LLM generates tokens autoregressively based on the previous tokens. Given a $\texttt{prompt}$ and the first $t-1$ generated response tokens $\texttt{res}^{1:t-1}$, it calculates a vector of *logits* $\mathop{\textsf{logit}}(\mathfrak e(\texttt{prompt}) \circ \mathfrak e(\texttt{res}^{1:t-1}))$, where $\mathfrak e$ is the embedding function and $\circ$ concatenates. The next token is then sampled from the vocab wrt

\[
    p^t
=   [p^t_1, ..., p^t_V]
=   \mathop{\textsf{softmax}}\bigl( \mathop{\textsf{logit}} \bigl( \mathfrak e(\texttt{prompt}) \circ \mathfrak e(\texttt{res}^{1:t-1}) \bigr) / T \bigr),
\]

where $T$ is the temperature. This sampling discards all information in the probability distribution. CIPHER retains it, but plugging back in not the sampled token, but a weighted average of (the embeddings of) all tokens. Formally,

\[
    \textstyle
    \mathfrak e^t
=   \sum_{i=1}^V
    p^t_i \mathfrak e(v_i)
\quad\textsf{where}\quad
    [p^t_1, ..., p^t_V]
=   \mathop{\textsf{softmax}}\bigl( \mathop{\textsf{logit}}\bigl( \mathfrak e(\texttt{prompt}) \circ \mathfrak e^{1:t-1} \bigr) / T \bigr).
\]

To emphasise, the embeddings $\mathfrak e(v_i)$ need only be calculated once. There is no $t$-th token which is to be embedded; rather the $t$-th embedding $\mathfrak e^t$ is calculated directly as a convex combination of the (precalculated) vocab embedding.

The generation process stops when either of two conditions hold.

1.  The end-of-sequence (EOS) token embedding becomes the nearest neighbour to the newly generated embedding.
2.  The maximal sequence length is reached.

If the resopnse length is $\tau$, the CIPHER response is $\texttt{cipher} = \mathfrak e^{1:t-1}$.


### Debate

The CIPHER debate procedure has the following steps.

1.  Convert the question and instructions into embeddings $\textsf{emb}_\texttt{prompt}$.

2.  For each debate round, form an embedding representation by concatenating $\textsf{emb}_\texttt{prompt}$ and (possible) CIPHER responses, $\texttt{cipher}_i$, from all debaters in previous rounds.

3.  Feed this embedding representation into the models without the token decoding step. The debaters generate refine CIPHER responses, $\texttt{cipher}_i$, following the previous procedure.

4.  To close the debate, convert the embedding responses back to natural language using nearest-neighbour search over the vocabulary set, then aggregate.

This is visualised in the paper, in Algorithm 1.

![CIPHER debate algorithm](attachments/CIPHER%20-%20Algorithm.png){ style="display: block; margin: 0 auto" }


## Results

CIPHER is benchmarked against three baselines.

1.  Single Answer: a single LLM provides one response in natural language.
2.  Self-Consistency: a single LLM independently generates multiple responses (five here), then applies majority voting.
3.  Natural Language Debate: each LLM provides an initial response, then uses each other's response to refine their previous response.

The third baseline is closest to CIPHER, with the primary difference being the method of communication (sampled token vs distribution).

Most experiments are conducted using LLaMA2-70B, one of the largest open-source models available at the time. The evaluation is across five reasoning benchmarks.

![Evaluation using same LLaMA model](attachments/CIPHER%20-%20Evaluation.png){ style="display: block; margin: 0 auto" }

Table 1 uses two debating LLMs, from the same family. Table 2 has LLaMA2-70B and LLaMA2-65B debate each other; unsurprisingly, the 70B version performs worse when paired with a 65B (vs another 70B) and the 65B does better with a 70B (vs 65B).

An ablation study is also undertaken, but not described here; see Section 5.3 of the paper for details.