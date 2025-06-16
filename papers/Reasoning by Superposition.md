---
tags:
    - reasoning
    - superposition
parent: 'Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought'
collections:
    - Reasoning
authors:
    - Zhu
    - Hao
    - Hu
    - Jiao
    - Russell
    - Tian
year: 2025
$version: 605
$libraryID: 1
$itemKey: RPTLH4JL

---
# Reasoning by Superposition Summary

*Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought*

*   2025-05; Zhu, Hao, Hu, Jiao, Russell, Tian

## High-Level Summary

*   Theoretical analysis of continuous chain of thought

*   Concretely, continuous CoT solves graph-reachability with diameter number of steps, rather than vertex squared—ie, *d* vs *O*(*n*^2^)

*   The continuous thought vector is a superposition enabling encoding of multiple search frontiers simultaneously—a 'parallel BFS'

*   Notably, this approach emerged *naturally*, without explicit supervision

*   Construction works for widely-used position encodings, not problem-specific ones

## Some Details

Graph Reachability Problem

*   Input:

    *   direct graph  $\mathcal G = (\mathcal V, \mathcal E)$  where  $\mathcal V = \{v_1, ..., v_n\}$  is the vocab

    *   root node  $r$  and two candidate destination nodes  $c_1$  and  $c_2$

*   Objective:

    *   determine which of  $c_1$  and  $c_2$  is reachable from  $r$

    *   it is given that precisely one is reachable

*   Edge  $e = (s, t) \in E \subseteq V^2$  is of the form  $(\textsf{source}, \textsf{target})$

> **Attention Chooser** (simplified)**.** If the current token is $\langle x \rangle$, then there is a construction of key-/query-matrices such that almost all attention is paid to position $i - \ell$, where $i$ is the current position and $\ell$ a pre-defined lag.

The core idea is to design the query and key vectors such that their inner product, which detemrines the attention score, is maximised at the desired position.

> **Superposition Construction** (simplified)**.** There is a choice of parameters such that the $c$-th continuous thought corresponds to the normalised, uniform superposition of (the embeddings of) all vertices reaching from the root $r$ within $c$ steps.

The proof involves constructing a two-layer transformer:

1.  copy source and target node embeddings of an edge to the token;
2.  perform one-step expansion of currently-explored vertices.

An MLP layer filters noise and equilibrates the weights remaining in the superposition.

## Empirical Validation

The theoretical claims are backed up with some empirical validation. The figure below compares the accuracy of Coconut with two layers (blue) with vanilla CoT (brown), CoT\* with 12 layers (green) and no CoT (pink).

![\<img alt="accuracy comparison: Coconut → 98%, CoT → 75%, CoT\* → 83%, baseline → 75%" data-attachment-key="PAM7IT58" width="280" height="359" src="attachments/PAM7IT58.png" ztype="zimage"> | 280](attachments/PAM7IT58.png){ style="display: block; margin: 0 auto" }
