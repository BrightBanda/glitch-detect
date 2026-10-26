# The Mechanics of AI Writing Detection: An Exhaustive Deep Dive

**A technical whitepaper on the mathematical models, linguistic signatures, and commercial architectures used to attribute authorship of machine-generated text**

| Field | Value |
| --- | --- |
| Document type | Technical whitepaper / research synthesis |
| Domain | AI safety, computational linguistics, NLP forensics |
| Version | 1.0 |
| Date | 26 October 2026 |
| Evidence base | Wikipedia (information theory, stylometry, watermarking, AI content detection), peer-reviewed NLP literature, arXiv preprints, and vendor technical documentation |
| Audience | Researchers, academic integrity officers, platform trust & safety engineers, editorial technologists |

> **Scope note.** This document is descriptive and analytical. It explains how detection systems compute their outputs, what those outputs do and do not mean, and where the underlying statistics break down. It does not provide operational guidance for evading detection; where evasion techniques are analysed, they are analysed as adversarial pressure on a measurement system, which is the standard framing in the security and robustness literature.

---

## Table of Contents

1. [Core Mathematical and Statistical Methodologies](#1-core-mathematical-and-statistical-methodologies)
   - [1.1 The generative premise](#11-the-generative-premise-why-statistics-can-reveal-authorship)
   - [1.2 Perplexity](#12-perplexity-the-mathematics-of-surprise)
   - [1.3 Burstiness](#13-burstiness-structural-variance-and-rhythmic-chaos)
   - [1.4 Advanced NLP concepts](#14-advanced-nlp-concepts)
   - [1.5 Cross-entropy loss](#15-cross-entropy-loss)
   - [1.6 From scores to decisions](#16-from-scores-to-decisions-calibration-thresholds-and-aggregation)
2. [Linguistic Footprints and "Clues" of AI Text](#2-linguistic-footprints-and-clues-of-ai-text)
3. [Deep Dive: Architectural Breakdown of the Gold Standards](#3-deep-dive-architectural-breakdown-of-the-gold-standards)
4. [Limitations, Obfuscation, and the False Positive Dilemma](#4-limitations-obfuscation-and-the-false-positive-dilemma)

---

## 1. Core Mathematical and Statistical Methodologies

### 1.1 The Generative Premise: Why Statistics Can Reveal Authorship

Every statistical detector rests on a single, surprisingly fragile premise: **human writing and machine writing occupy different regions of a probability distribution over token sequences.** The detector's job is to estimate which region a document came from.

A causal language model defines a probability distribution over sequences by the chain rule of probability:

```text
p(x_1, x_2, ..., x_N)  =  ∏_{t=1}^{N} p(x_t | x_1, ..., x_{t-1})  =  ∏_{t=1}^{N} p(x_t | x_<t)
```

This factorisation is the entire mathematical foundation of autoregressive generation. A model is trained by minimising the negative log-likelihood of human text under this factorisation; it then generates by sampling from its own conditional distributions. That asymmetry — *trained on human text, sampled from itself* — is what leaks information about provenance.

Formally, a detector implements a hypothesis test:

```text
H_human : x ~ P_human
H_AI    : x ~ P_AI
```

The optimal test, by the Neyman–Pearson lemma, is a likelihood ratio threshold:

```text
Λ(x) = log [ P_AI(x) / P_human(x) ]  >  τ    ⇒  classify as AI
```

In practice, `P_human` is unknown and enormous (it is the distribution of all human writing, past and future, across every genre and idiolect), while `P_AI` is unknown in a different way (it is a mixture over hundreds of models, versions, prompts, temperatures, and fine-tunes). Neither distribution is estimable in closed form. Every commercial detector is therefore an *approximation* of this ratio, and every one of its failure modes traces back to one of those two approximations.

Three consequences follow immediately, and they recur throughout this document:

1. **The alternative hypothesis is composite and non-stationary.** "AI text" is not one author but a moving mixture. A detector calibrated against GPT-3.5 in 2023 is testing a hypothesis that no longer exists in 2026.
2. **The null hypothesis is heterogeneous.** "Human text" spans tweets, legal contracts, poetry, lab reports, and the writing of second-language learners. Any single threshold imposes a false-positive cost that is unevenly distributed across those subpopulations.
3. **The decision is a trade-off, not a measurement.** Detectors do not output "authorship." They output a statistic that is compared against a threshold chosen by a vendor, and that threshold encodes a policy preference about which error is worse.

That last point is why the industry reports **TPR at a fixed FPR** (true-positive rate at, say, a 1% or 0.01% false-positive rate) rather than "accuracy." Accuracy is a property of a dataset; TPR-at-FPR is a property of an operating point, which is what a deployed system actually has.

```text
TPR = TP / (TP + FN)          sensitivity / recall
FPR = FP / (FP + TN)          fall-out
PPV = TP / (TP + FP)          precision  (depends on prevalence — see §4.1)
F1  = 2 · (PPV · TPR) / (PPV + TPR)
```

---

### 1.2 Perplexity: The Mathematics of Surprise

#### 1.2.1 Definition

Perplexity is an information-theoretic measure of how well a probability distribution predicts a sample. For a discrete distribution `p`, [Wikipedia's definition](https://en.wikipedia.org/wiki/Perplexity) is:

```text
PP(p) := 2^{H(p)} = 2^{ -∑_x p(x) log₂ p(x) } = ∏_x p(x)^{-p(x)}
```

where `H(p)` is the Shannon entropy of the distribution in bits. Perplexity is the exponentiation of entropy: it converts an average bit-count into an interpretable "effective number of choices."

The canonical intuition is the fair *k*-sided die. A uniform distribution over `k` outcomes has entropy `log₂ k` and therefore perplexity `k`. A random variable with perplexity `k` carries the same uncertainty as a fair `k`-sided die — one is "*k*-ways perplexed" about its value. This is why perplexity is often described as a **branching factor**: a model with perplexity 21.4 is, on average, as uncertain as if it were choosing uniformly among ~21 equally likely next tokens.

Applied to a model rather than a distribution, perplexity of a model `q` on a test sample `x_1 … x_N` is:

```text
PP(q) = b^{ -(1/N) ∑_{i=1}^{N} log_b q(x_i) } = ( ∏_{i=1}^{N} q(x_i) )^{-1/N}
```

with `b` conventionally 2. Note the exponent: the average negative log-probability. That quantity *is* the cross-entropy of the empirical distribution with respect to `q`. Therefore:

```text
PP(q) = exp( H(p̃, q) )        and, for language models,
PPL   = exp( -(1/N) ∑_{t=1}^{N} log p(x_t | x_<t) )
```

#### 1.2.2 Worked example

Consider the seven-token sentence *"The cat sat on the mat."* and a hypothetical model's conditional probabilities:

| Token | p(x_t \| x_&lt;t) | Surprisal −log₂ p (bits) |
| --- | --- | --- |
| `The` | 0.050 | 4.32 |
| ` cat` | 0.020 | 5.64 |
| ` sat` | 0.030 | 5.06 |
| ` on` | 0.080 | 3.64 |
| ` the` | 0.100 | 3.32 |
| ` mat` | 0.010 | 6.64 |
| `.` | 0.200 | 2.32 |
| **Total** | — | **30.94 bits** |

```text
Cross-entropy = 30.94 / 7 = 4.42 bits per token
Perplexity    = 2^{4.42} ≈ 21.4
```

The model is "21.4-ways perplexed" per token. If the same sentence were scored by a model that assigned every token probability 0.9, perplexity would fall to ~1.11 — the model would be nearly certain, and the text would look maximally machine-like under that model.

#### 1.2.3 Why low perplexity signals machine generation

Generation and detection are mirror images of the same computation. When a model samples `x_t ~ p(·|x_<t)`, the *expected* surprisal of the sampled token equals the entropy of that conditional distribution. A model sampling from itself therefore produces text whose average surprisal converges to its own conditional entropy — the **minimum achievable cross-entropy for that model on that context**. Human text, by contrast, is drawn from a different distribution; it routinely contains collocations, rhetorical moves, and lexical choices that the model assigns low probability. Its cross-entropy is higher.

This gives the classical detector score:

```text
score(x) = PPL_model(x)          low  ⇒  machine-like
score(x) = CE_model(x)           low  ⇒  machine-like
```

The corollary is uncomfortable: **perplexity is not a property of text, it is a property of text relative to a model.** A text is "low perplexity" only under some `q`. Change `q`, and the score changes — sometimes drastically. GPT-2's perplexity ranking of a document has little relation to GPT-5's.

#### 1.2.4 The empirical anchor

The classic benchmark figure, cited on Wikipedia from [Brown et al. (1992)](https://aclanthology.org/J92-1003/), is that the lowest published perplexity on the Brown Corpus (1 million words of American English, mixed genres) as of 1992 was **≈247 per word**, corresponding to 7.95 bits per word with a trigram model. Two lessons survive from that result:

- **Perplexity is genre-sensitive.** "It is often possible to achieve lower perplexity on more specialised corpora, as they are more predictable." Legal boilerplate, API documentation, and methodological prose are *intrinsically* low-perplexity. This is the root of the technical-writing false positive (§4.3).
- **Perplexity is not calibrated to accuracy.** Even in 1992 it was known that "simply guessing that the next word is *the*" achieves 7% accuracy on Brown, not `1/247`. Perplexity measures uncertainty, not correctness, and later work found it a poor predictor of downstream task performance (Klakow & Peters, 2002, on word error rate; Miaschi et al., 2021, on linguistic features driving perplexity).

#### 1.2.5 Where perplexity fails as a detector

| Failure mode | Mechanism | Consequence |
| --- | --- | --- |
| Model-relativity | Score depends entirely on the reference LM | A detector built on GPT-2 cannot rank GPT-5 text reliably |
| Genre confounding | Formulaic genres have low entropy | Contracts, specs, methods sections score "AI" |
| Repetition exploit | Repeated n-grams have high probability | Degenerate or padded text scores "AI" |
| Domain mismatch | Technical vocabulary is low-probability under a general LM | Specialist human writing scores "human" (false negative) |
| Paraphrase laundering | Rewriting raises perplexity under the original model | Detection collapses after one paraphrase pass |
| RLHF drift | Instruction-tuned models are not the pretraining distribution | The "machine region" moves with each release |

This is precisely why no serious commercial detector in 2026 ships raw perplexity as its decision variable. Turnitin's documentation is explicit on this point: their model "is not explicitly programmed to evaluate specific signals such as 'burstiness,' 'perplexity,' or other individual metrics sometimes referenced in public discussions. Instead, it learns statistical patterns from our training data." Perplexity survives in modern systems as *one learned feature among thousands*, not as the classifier itself.

---

### 1.3 Burstiness: Structural Variance and Rhythmic Chaos

#### 1.3.1 Definition

If perplexity measures the *mean* level of surprise, burstiness measures its *dispersion*. A bursty process alternates between long quiet periods and short violent ones; a non-bursty process ticks along at a constant rate. Human prose is bursty at every scale — within a sentence (a clause stack followed by a three-word fragment), across sentences (a 42-word sentence followed by "No."), across paragraphs, and across documents.

The standard scalar formalisation, from Goh & Barabási's work on bursty human dynamics, is:

```text
B = (σ - μ) / (σ + μ)          B ∈ (-1, 1)
```

where `μ` and `σ` are the mean and standard deviation of the inter-event interval (here: sentence lengths, or gaps between high-surprisal tokens). `B = 0` is Poisson-like (memoryless, "random"); `B → 1` is maximally bursty; `B → -1` is perfectly regular. Equivalent descriptors used in stylometry include the coefficient of variation `CV = σ/μ`, the Fano factor `F = σ²/μ`, the Gini coefficient of sentence lengths, and the Allan factor for clustered point processes.

At the token level, burstiness becomes the variance of per-token surprisal, or the autocorrelation of `−log p(x_t)` across positions:

```text
S_t   = -log p(x_t | x_<t)
B_tok = Var(S_t) / E[S_t]                 (Fano factor of surprisal)
ρ(k)  = Corr(S_t, S_{t+k})                (surprisal autocorrelation at lag k)
```

#### 1.3.2 The human distribution

Human writing has a heavy-tailed sentence-length distribution and a lumpy surprisal profile. Empirically, in unedited academic and journalistic prose:

- Sentence lengths typically span roughly 3–60 words with a right-skewed distribution and a standard deviation often in the 8–14 word range.
- Surprisal spikes at proper nouns, jargon, numerals, and clause boundaries, then collapses across function-word runs.
- Paragraph lengths vary with rhetorical function: a one-line paragraph for emphasis, a 200-word block for narrative.
- There are **discontinuities**: digressions, self-corrections ("— or rather, what I mean is —"), asides in parentheses, sentence fragments used deliberately.

#### 1.3.3 The machine distribution

A model sampling from its own distribution with temperature `T` and nucleus/top-k truncation produces a *contracted* distribution. Three mechanisms flatten variance:

1. **Mode-seeking decoding.** Low temperature and top-p truncation remove the tails. The rare, surprising token — the one that creates a burst — is exactly the token that gets truncated away.
2. **RLHF preference collapse.** Preference optimisation rewards prose that "reads well" to annotators, which correlates with balanced, mid-length, evenly-paced sentences. The reward model is a variance *penaliser*.
3. **Averaging over an essay.** A 600-word response is generated as a single trajectory; the model has no global plan that would motivate a deliberate three-word sentence for rhythm. Local coherence dominates.

The observable result is the famous "18-to-24-word band": sentence lengths cluster tightly around ~20 words with a standard deviation often half that of a matched human sample; paragraph lengths are near-uniform; every paragraph has a topic sentence, three supports, and a transition.

#### 1.3.4 Illustrative contrast

```text
HUMAN (bursty)                      |  MACHINE (non-bursty)
------------------------------------|------------------------------------------
"Forty-two. That's the number."     |  "There are several factors to consider
                                    |   when examining this issue."
Then a 58-word sentence that       |
wanders, doubles back, and ends    |  "First, it is important to understand
with a clause the writer clearly   |   the context. Second, the relevant
did not plan when starting it.     |   variables must be identified. Third,
                                    |   a balanced assessment follows."
```

The right-hand column is not *wrong*; it is *uniform*. Every sentence is a complete, well-formed, medium-length declarative. There is no risk, no rhythm, and — critically — no variance for a statistical test to detect as human.

#### 1.3.5 Why burstiness is a weaker signal than it is sold as

"Perplexity and burstiness" is the most repeated phrase in consumer-facing detector marketing, largely inherited from GPTZero's original 2023 heuristic. It is worth being precise about its limits:

- **It is trivially gameable.** Alternating sentence lengths is a one-line prompt instruction ("vary your sentence length; include one very short sentence per paragraph"). Detectors that rely on sentence-length variance alone fail immediately.
- **It is genre-dependent.** Legal writing, technical documentation, and controlled-language standards (e.g. Simplified Technical English) are *deliberately* non-bursty. So is a well-edited newspaper brief.
- **It measures variance, not authorship.** A human editing for uniformity and a machine generating for uniformity produce the same statistic. The detector cannot distinguish the cause.
- **The estimator is noisy on short text.** With 8 sentences, a sample standard deviation has enormous sampling error; the confidence interval on `B` swamps the human/machine gap.

Modern systems therefore treat burstiness as one feature in a learned representation rather than a decision rule — which is exactly what Turnitin describes and what GPTZero's architecture (§3.2) implements.

---

### 1.4 Advanced NLP Concepts

#### 1.4.1 N-gram frequency analysis and rank-based tests

An n-gram language model estimates `p(x_t | x_{t-n+1..t-1})` by relative frequency with smoothing (add-k, Kneser–Ney). Before neural LMs, perplexity under an n-gram model was *the* detection score. The modern descendant is **rank-based** rather than probability-based, and it is considerably more robust.

For each token, compute its rank `r_t` in the model's next-token distribution (rank 1 = most likely). Then bucket:

```text
Test 1:  fraction of tokens with r_t ≤ 10       ("top-10 rate")
Test 2:  fraction of tokens with r_t ≤ 100
Test 3:  fraction of tokens with r_t ≤ 1000
```

This is the core of [**GLTR**](https://arxiv.org/abs/1906.04043) (Gehrmann, Strobelt & Rush, 2019 — the "Giant Language model Test Room"). GLTR renders each token as a coloured chip — green (top-10), yellow (top-100), red (top-1000), purple (beyond) — producing the instantly recognisable "mostly green" signature of machine text. Human text has a characteristic mixture: a green core of function words plus substantial yellow/red/purple mass at content words, names, and idiom.

The advantage over raw perplexity is that rank statistics are **monotone-invariant**: they do not depend on the model's calibration, only on its ordering. A badly calibrated model still ranks well. This is why GLTR-style features generalise across model families better than perplexity does.

Two derived measures matter:

```text
Entropy of the next-token distribution:   H_t = -∑_v p(v) log p(v)
Predictive uncertainty profile:           mean(H_t), var(H_t), autocorr(H_t)
```

Human text tends to show *high entropy at content-word positions and low entropy at function-word positions* — a wide dynamic range. Machine text shows a compressed dynamic range: the model is never very uncertain, because it chose the token it was most confident about.

#### 1.4.2 Vector space embeddings

An embedding maps a token, sentence, or document into `R^d` such that semantically or stylistically similar items are close. Three generations are relevant:

| Generation | Example | Unit | Use in detection |
| --- | --- | --- | --- |
| Static | word2vec, GloVe, fastText | word type | Lexical-semantic drift, vocabulary coverage |
| Contextual | BERT, RoBERTa, DeBERTa | token in context | The backbone of modern classifiers; `[CLS]` pooled as a document vector |
| Sentence/document | SBERT, E5, proprietary | span | Redundancy measurement; retrieval; clustering |

The decisive property for detection is that a transformer's pooled representation encodes **style as well as content**. Two texts about the same topic, written by a human and a machine, land at different points because the representation disentangles — imperfectly — *what is said* from *how it is said*. A linear probe on top of a frozen RoBERTa `[CLS]` vector already separates human from ChatGPT text well above chance; a fine-tuned classifier does substantially better.

Three embedding-based signals recur in the literature:

1. **Centroid distance.** Distance from a document embedding to the human and machine centroids of a reference corpus.
2. **Inter-sentence coherence/redundancy.** Mean cosine similarity between consecutive sentence embeddings. Machine text often sits in a narrow band of high mutual similarity (§2.4).
3. **Intrinsic dimensionality.** [Tulchinskii et al. (2023)](https://arxiv.org/abs/2306.04723) showed that the *intrinsic dimension* of the contextual-embedding manifold traversed by a text is systematically lower for machine-generated text. The estimator (a Petrov–Tulchinskii–Kushnareva-style MLE over k-nearest-neighbour distance ratios) requires no training data and is comparatively robust to paraphrase — one of the few genuinely model-agnostic geometric signals.

A caution that applies to all embedding methods: **anisotropy**. Contextual embeddings occupy a narrow cone, so raw cosine similarities are compressed into a high baseline and small stylistic differences are easy to drown. Whitening, length normalisation, or contrastive fine-tuning is usually necessary before cosine distances become discriminative.

#### 1.4.3 Entropy and information-theoretic measures

Beyond perplexity, four quantities are load-bearing:

```text
Shannon entropy:        H(X)      = -∑ p(x) log p(x)
Conditional entropy:    H(X|Y)    = -∑ p(x,y) log p(x|y)
Mutual information:     I(X;Y)    = H(X) - H(X|Y)
KL divergence:          D_KL(P‖Q) = ∑ p(x) log [p(x)/q(x)]
Jensen–Shannon:         JSD(P‖Q)  = ½D_KL(P‖M) + ½D_KL(Q‖M),  M = ½(P+Q)
```

- **Conditional entropy of the text under the model** is the average surprisal; its exponent is perplexity. Low conditional entropy ⇒ high predictability ⇒ machine-leaning.
- **Predictive entropy of the model** (`H_t`, the entropy of the next-token distribution) is a different quantity and is often *more* informative than surprisal. Machine text is generated at positions where `H_t` is moderate and the sampled token is near the mode; human text contains positions where `H_t` is high and the token is nevertheless chosen.
- **KL divergence between the document's empirical unigram/bigram distribution and a human reference distribution** is a classic stylometric distance. It is cheap, interpretable, and — importantly — *not* dependent on access to the generator.
- **Compression.** Since `PPL = exp(CE)` and cross-entropy lower-bounds achievable code length, "how well does this text compress under a model trained on human text?" is a detection score. Practical variants use `gzip`/`bz2`/PPM as model-free compressors: machine text, being more predictable, compresses slightly better. The effect size is small and confounded by genre, but compression-based classification has the advantage of requiring no neural model at all.

The information-theoretic view also supplies the **upper bound on detectability**. If `P_human` and `P_AI` overlap, no test can separate them perfectly. Quantitatively, for any detector:

```text
FPR + FNR  ≥  1 - d_TV(P_human, P_AI)
```

where `d_TV` is the total variation distance. This is Le Cam's bound, and [Sadasivan et al. (2023)](https://arxiv.org/abs/2303.11156) apply exactly this argument to AI-text detection: as language models are trained to minimise the divergence between their output distribution and human text, `d_TV → 0` and every detector's error rate must approach that of a coin flip. The bound is not a limitation of current engineering; it is a property of the problem.

#### 1.4.4 Log-probability distributions and curvature tests

The most elegant family of detectors exploits a geometric property of machine text: **it sits near a local maximum of the model's log-likelihood surface.**

[**DetectGPT**](https://arxiv.org/abs/2301.11305) (Mitchell, Lee, Khazatsky, Manning & Finn, 2023) makes this explicit. For a passage `x` and a perturbation function `q` (mask-and-fill with T5, producing `x̃`), define:

```text
d(x, p_θ, q) = log p_θ(x) - E_{x̃ ~ q(·|x)} [ log p_θ(x̃) ]
```

The claim is that machine-generated text has *negative* curvature — perturbing it tends to *reduce* log-probability, because the original sits at a peak. Human text sits off-peak, so random perturbations land at similar or higher likelihood. The detector thresholds `d(x)`. DetectGPT reported roughly 0.95 AUROC on the XSum fake-news detection setting, an improvement of about 0.1 AUROC over the prior zero-shot baseline, at the cost of ~100 model calls per document.

[**Fast-DetectGPT**](https://arxiv.org/abs/2310.05130) (Bao et al., 2023) removes the sampling loop. It observes that the expectation over perturbations can be approximated analytically by a **conditional probability curvature** using a single model and a single forward pass:

```text
d(x) ≈ [ log p_θ(x) - μ̃ ] / σ̃
```

where `μ̃` and `σ̃` are the mean and standard deviation of the log-probability under the model's own conditional sampling at each position. This is roughly 340× cheaper and, in the authors' evaluation, more accurate than the sampling-based original.

[**Binoculars**](https://arxiv.org/abs/2401.12070) (Hans et al., 2024) takes a different tack: contrast two closely related models, an *observer* and a *performer*.

```text
B(x) = log PPL_observer(x) / log X-PPL(x)
```

where cross-perplexity `X-PPL` measures how surprised the observer is by the performer's distribution at each position:

```text
X-PPL = exp( -(1/N) ∑_t  ∑_v p_performer(v | x_<t) · log p_observer(v | x_<t) )
```

The ratio cancels topic difficulty. A document that is "easy" because it is about a familiar topic raises both terms; a document that is "easy" because it was generated by a model like the performer raises only the numerator. Binoculars reported >90% detection of ChatGPT-family output at a **0.01%** false-positive rate, with no training data — the strongest published zero-shot result of its generation.

| Method | Signal | Models needed | Training data | Reported operating point | Principal weakness |
| --- | --- | --- | --- | --- | --- |
| Perplexity threshold | `CE_model(x)` | 1 | none | Highly variable | Model- and genre-relative |
| GLTR / rank tests | token rank histogram | 1 | none (reference corpus optional) | Qualitative + AUC | Surface-level; prompt-sensitive |
| DetectGPT | log-prob curvature | 1 (+ perturbation model) | none | ~0.95 AUROC (XSum) | ~100 forward passes/doc |
| Fast-DetectGPT | conditional curvature | 1 | none | Beats DetectGPT at ~1/340 cost | Needs white-box logits |
| Binoculars | PPL / X-PPL ratio | 2 | none | >90% TPR @ 0.01% FPR | Needs two matched open models |
| Intrinsic dimension | embedding geometry | 1 encoder | none | AUC ~0.95+ on benchmarks | Sensitive to encoder choice |
| Supervised classifier | learned features | 1 encoder | millions of docs | 99%+ in-distribution | Fails OOD; needs retraining |

---

### 1.5 Cross-Entropy Loss

#### 1.5.1 Definition and relation to perplexity

Cross-entropy measures the cost of encoding samples from a true distribution `p` using a model `q`:

```text
H(p, q) = -∑_x p(x) log q(x)
```

For a single observed sequence, the empirical cross-entropy is the average negative log-likelihood:

```text
CE = -(1/N) ∑_{t=1}^{N} log q(x_t | x_<t)
```

Two identities do all the work:

```text
CE(p, q) = H(p) + D_KL(p ‖ q)
PPL      = exp(CE)
```

The first says that cross-entropy decomposes into the irreducible entropy of the true distribution plus the penalty for using the wrong model. The second says perplexity is just exponentiated loss. **A detector that thresholds cross-entropy is thresholding perplexity; the two framings are mathematically identical.** Where they differ is in engineering: cross-entropy is what the training loop computes, so it is the natural interface for a learned detector.

#### 1.5.2 Why detectors use loss values rather than probabilities

There are four practical reasons the loss formulation dominates:

1. **Numerical stability.** Log-probabilities are additive over thousands of tokens; raw probabilities underflow to zero within a paragraph.
2. **Length normalisation is explicit.** Dividing by `N` makes documents of different lengths comparable — though, as §4.3 shows, this normalisation is itself a source of bias.
3. **Calibration is tractable.** A scalar score can be calibrated with isotonic regression or temperature scaling against a labelled human corpus, producing an interpretable threshold at a target FPR.
4. **It aligns with the generator's own objective.** Both the generator and the detector are computing `−log p`; the detector is, in effect, asking "how well does this text minimise the objective that produced it?"

#### 1.5.3 The "loss alignment" hypothesis and the MDL framing

The hypothesis is simple: text produced by minimising `−log p` will exhibit loss values near the model's training floor, while human text will exhibit higher loss because humans optimise for communication, not for the model's likelihood.

This is a **minimum description length** (MDL) argument. The best explanation of a string is the shortest program that generates it. If a language model is a good model of English, then text that the model can encode cheaply is text that is "explained" by the model — and machine text is, by construction, the output of the model's own best explanation of itself. Human text carries information the model did not put there.

The hypothesis has a precise failure mode: **the model that generated the text is not the model that scores it.** If a document was written by GPT-5 and scored by Llama-3, the loss is not near the floor; it is near the cross-entropy between two different distributions. This "detector–generator gap" is why white-box detectors (which query the actual generator) outperform black-box ones, and why the strongest commercial systems train supervised classifiers on outputs from *many* generators rather than relying on any single model's loss.

#### 1.5.4 Reference implementation

```python
import torch

@torch.no_grad()
def cross_entropy_and_perplexity(model, tokenizer, text, max_length=2048):
    """Average token-level cross-entropy (nats) and perplexity for a passage.

    Uses a causal LM's own next-token objective:
        CE  = -(1/N) * sum_t log p(x_t | x_<t)
        PPL = exp(CE)
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length)
    input_ids = enc["input_ids"]

    out = model(input_ids=input_ids, labels=input_ids)
    # HF shifts labels internally: loss is already mean over predicted tokens
    ce_nats = out.loss.item()

    n_scored = input_ids.shape[1] - 1          # first token has no predecessor
    total_nll = ce_nats * n_scored             # recover the sum, not the mean

    return {
        "ce_nats_per_token": ce_nats,
        "ce_bits_per_token": ce_nats / 0.6931471805599453,
        "perplexity": float(torch.exp(torch.tensor(ce_nats))),
        "n_scored_tokens": n_scored,
        "total_nll_nats": total_nll,
    }
```

Two implementation details that matter more than they look:

- **Length normalisation is mandatory but biased.** A 40-token document has a high-variance mean; a 4,000-token document has a stable one. Binoculars and most production systems require a minimum token count (50–300) before returning a score.
- **Tokenisation changes the denominator.** Byte-pair tokenisers compress common words, so "per-token" cross-entropy is not comparable across tokenisers. Perplexity comparisons across models with different vocabularies are, strictly speaking, meaningless — a point [Wikipedia's article](https://en.wikipedia.org/wiki/Perplexity) flags indirectly by noting perplexity's sensitivity to linguistic features and sentence length.

#### 1.5.5 Calibration: turning loss into a decision

A raw CE score is not a probability of authorship. Production systems map it through a calibration function fitted on held-out human and machine corpora:

```text
p_AI(x) = σ( a · CE(x) + b )                      (Platt scaling)
p_AI(x) = g( CE(x) )                              (isotonic regression, monotone g)
threshold τ chosen s.t.  FPR(τ) = α               (e.g. α = 0.01 or 0.0001)
```

Post-hoc remapping is not cosmetic. GPTZero's paper describes a remapping function `r: R³ → R³` applied to the three class probabilities specifically to "improve calibration and reduce false-positive predictions," and documents an appendix devoted to "Postprocessing Predictions to Limit False Positives." Pangram describes averaging and smoothing token-level scores explicitly "to discourage spurious flickering." Turnitin suppresses all document scores below 20% and reports only an asterisk. In every case, the published number a user sees is the output of a **policy layer**, not a raw statistical estimate.

---

### 1.6 From Scores to Decisions: Calibration, Thresholds, and Aggregation

#### 1.6.1 The aggregation problem

A document is not a token. Detectors must reduce thousands of token scores to one number, and the choice of reducer is a design decision with large consequences:

| Aggregator | Behaviour | Typical use |
| --- | --- | --- |
| Mean of token/segment scores | Stable, dilutes short AI spans | Whole-document probability |
| Median | Robust to outliers | Noisy OCR / mixed formatting |
| Max / top-k quantile | Sensitive to any AI span | "Did AI touch this at all?" |
| Fraction above threshold | Interpretable as "% of sentences" | Turnitin's document score |
| Windowed mean + overlap | Localises boundaries | Pangram 4, GPTZero |
| Learned pooling / attention | Best in-distribution | End-to-end classifiers |

Turnitin's documentation is unusually explicit: the model assigns each sentence a score in `[0,1]` "based on the sentence in context with the surrounding text," sentences above a threshold are labelled AI, and "the document level AIW score is the percentage of sentences that are above the threshold." That is a *fraction-above-threshold* aggregator, not a mean.

The semantic consequence is frequently misread: **a "50%" Turnitin score does not mean half the words are AI.** It means half of the *qualifying prose sentences* crossed the threshold. Because qualifying sentences exclude lists, headings, references, tables, code, and formulae, the denominator is not the document.

#### 1.6.2 Why the base rate dominates

The single most important — and most frequently omitted — calculation in applied detection is Bayes' theorem. Let `π` be the prevalence of AI-assisted documents in the population being screened, `TPR` the detector's sensitivity, and `FPR` its false-positive rate. Then:

```text
P(AI | flagged) = (π · TPR) / (π · TPR + (1 - π) · FPR)
```

| Prevalence π | TPR | FPR | Documents screened | Humans wrongly flagged | PPV (precision) |
| --- | --- | --- | --- | --- | --- |
| 10% | 90% | 1% | 100,000 | 900 | **90.9%** |
| 1% | 90% | 1% | 100,000 | 990 | **47.6%** |
| 1% | 99.66% | 0.0041% | 100,000 | 4 | **99.96%** |
| 0.1% | 90% | 1% | 100,000 | 999 | **8.3%** |
| 10% | 90% | 5% | 100,000 | 4,500 | **66.7%** |

Read the second and third rows together. Two detectors can both be described as "99% accurate" while one produces a flag that is right half the time and the other produces a flag that is right 99.96% of the time. The difference is not accuracy; it is the false-positive rate *at the operating point* combined with the population being screened. This is why Pangram's headline metric is 0.0041% FPR rather than "99.66% accuracy," and why Originality.ai's own documentation warns that "the rate of false positives (even if low) is still too high to be relied upon for disciplinary action."

#### 1.6.3 Ensemble and adversarial training

The final layer of a production detector is usually an ensemble: a supervised transformer head, a perplexity/curvature feature, stylometric features, and heuristic artifact detectors, combined by a shallow meta-classifier or a fixed weighted vote. Adversarial training — augmenting the training set with paraphrased, humanised, and translation-laundered examples — is the standard defence and is described in detail by both GPTZero (four-tier red teaming) and Pangram (hard-negative mining). The trade-off is documented in GPTZero's limitations section: adversarial training "is known to trade-off performance on in-distribution examples" (citing Tramèr & Boneh, 2019; Raghunathan et al., 2019).

---

## 2. Linguistic Footprints and "Clues" of AI Text

Section 1 described the machinery. This section describes the phenomenology — what actually differs in the text, why it differs, and how much evidential weight each difference can bear.

The organising claim is that **clues are not equal, and their inequality is structural rather than empirical.** A clue is useful to a detector in proportion to how deeply it is coupled to the generative process. Surface lexicon is shallowly coupled: it is steerable by a single instruction ("avoid the word *delve*") and it drifts with every model release. Discourse architecture and referential grounding are deeply coupled: they are consequences of the objective function and of the absence of a communicative situation, and no amount of synonym substitution removes them. This distinction explains the otherwise puzzling empirical pattern in which detectors that rely on word lists collapse under paraphrasing while detectors that model sentence-to-sentence information flow degrade gracefully.

It also explains why the public conversation about detection is dominated by the weakest evidence. Lexical tells are the only clues a human reader can see without instrumentation, so they are the ones that get written about, argued over, and mocked. They are also the ones a vendor can least afford to weight heavily.

---

### 2.1 Four Tiers of Evidence

Detection signals sort into four tiers, ordered by depth of coupling to the production process.

| Tier | Signal class | Representative signal | Coupled to | Survives paraphrasing? | Evidential weight |
| --- | --- | --- | --- | --- | --- |
| **T0** | Artifactual / provenance | Chatbot boilerplate, markdown residue, C2PA metadata, watermark | The *delivery pipeline*, not the language | Yes (untouched) / No (stripped) | Near-certain when present; near-zero recall |
| **T1** | Distributional | Perplexity, log-rank, curvature, token entropy | The model's conditional distribution | Partially — paraphrase raises perplexity toward human range | Moderate; degrades under rewriting |
| **T2** | Morphosyntactic / discourse | Nominalisation rate, participial clauses, sentence-length variance, connective density, section symmetry | Instruction tuning and plan-then-write generation | **Largely survives** | High; the most stable statistical channel |
| **T3** | Pragmatic / referential | Absence of verifiable particulars, entailment closure, symmetrical hedging, missing stance | The absence of a communicative goal | **Survives**; paraphrasing cannot invent a referent | Highest in principle; hardest to operationalise |

The tiers are not a ranking of accuracy but of *adversarial robustness*. A T1 detector can be defeated by a thesaurus. A T2 detector can be defeated only by rewriting the argument's shape, which is most of the work of writing. A T3 detector can be defeated only by supplying the missing referents — that is, by actually doing the reporting. This is the analytical core of why "AI humanisers" underperform their marketing: they operate almost entirely on T0 and T1, while the durable signal lives in T2 and T3.

There is a statistical trap here that deserves naming, because it recurs in every public dispute about a flagged document. A clue's diagnostic value is its **likelihood ratio**, not its probability under the AI hypothesis:

```text
LR(c) = P(c | AI) / P(c | human)

Posterior odds = Prior odds × LR(c)
```

A construction can be extremely common in AI text — `P(c | AI) = 0.4` — and still be weak evidence, because it is also common in human text. "Delve" is the canonical example: it is genuinely over-represented in post-2022 machine-written academic prose, and it is also a perfectly ordinary word used by tens of thousands of human writers, particularly in West African, South Asian, and postcolonial academic Englishes where it is standard register. A detector that treats `P(c | AI)` as if it were `P(AI | c)` is committing the prosecutor's fallacy with a dictionary.

---

### 2.2 Vocabulary Bias: The LLM Lexicon

#### 2.2.1 Mechanism

Three forces push model output toward a narrow, recognisable lexicon.

**Preference optimisation selects for "professional" register.** RLHF and DPO reward responses that human raters judge helpful, polished, and appropriately formal. Raters systematically prefer prose that sounds like a well-edited report. The optimiser therefore discovers a lexical basin: abstract nouns, evaluative adjectives, and verbs of exposition. The result is not a random sample from the pretraining distribution but the *mode* of a preference-weighted distribution — "edited English" with the idiolect removed.

**Instruction tuning produces noun-heavy, informationally dense prose.** This is measurable and has been measured. [Reinhart et al. (PNAS, 2025)](https://arxiv.org/abs/2410.16107), using Douglas Biber's 66-feature tagset on parallel human/LLM corpora built from Llama 3 and GPT-4o variants, found that instruction-tuned models used **present participial clauses at 2 to 5 times the human rate**, **nominalisations at 1.5 to 2 times the human rate**, and **agentless passives at roughly half the human rate**. Critically, the *base* models matched human rates more closely than the instruction-tuned ones. The authors conclude that instruction tuning "trains them in a particular informationally dense, noun-heavy style" rather than teaching them to write more like people. This is a strong claim about causation, and it inverts the intuition that better models are harder to detect: the alignment layer *adds* a fingerprint.

**Boilerplate contamination.** A large fraction of pretraining text is corporate, institutional, and academic prose that already contained these habits. The model amplifies an existing register rather than inventing one — which is why the LLM lexicon overlaps heavily with the vocabulary of management consultancy and grant applications.

#### 2.2.2 Measuring excess vocabulary

The strongest empirical work on lexical shift does not classify documents at all. It estimates population-level prevalence from word-frequency discontinuities. [Kobak et al. (Science Advances, 2025)](https://arxiv.org/abs/2406.07016) analysed over **15 million PubMed abstracts (2010–2024)** and computed, for each word, the change in frequency across the ChatGPT release boundary. Under a model in which affected documents contain the word at an elevated rate and unaffected documents contain it at baseline, the observed frequency increase bounds the affected fraction from below:

```text
Δ(w)  =  f_post(w) − f_pre(w)                       (excess frequency per token)
Δ(w)  ≈  q · E[count of w | affected]  −  dilution
⇒  q  ≥  Δ(w) / max plausible count per document
```

Because the bound is computed per word and then maximised over a curated set of "excess" words with tight confidence intervals, it yields a *lower bound* on prevalence that does not depend on any classifier. Their estimate: **at least 13.5% of 2024 abstracts were LLM-processed**, reaching **40% in some subcorpora** by discipline, country, and journal. They note the effect exceeds that of major world events, including the COVID-19 pandemic, on biomedical vocabulary. A parallel study of **950,965 papers** on arXiv, bioRxiv, and Nature portfolio journals by [Liang et al. (2024)](https://arxiv.org/abs/2404.01268) estimated up to **17.5%** LLM-modification in computer science and up to **6.3%** in mathematics and the Nature portfolio.

The methodological point is important for understanding commercial detectors: **population-level estimation and document-level attribution are different problems with different statistical properties.** Kobak et al. explicitly prefer corpus-level inference because it is robust where individual-instance inference is not. A vendor selling a per-document score is solving the harder problem, and the base-rate arithmetic of §1.6.2 tells you how much harder.

#### 2.2.3 The lexeme inventory

The following constructions are the most frequently documented. The third column is the analytically interesting one: each item exists because it solves a local generation problem.

| Lexeme / construction | Typical slot | Functional motive | Persistence |
| --- | --- | --- | --- |
| *delve into* | "This paper delves into…" | Avoids the plain verb *examine*; signals depth without asserting method | High in academic prose; decaying in chat output |
| *underscore* / *highlight* | "These findings underscore…" | Converts a result into a significance claim without specifying the mechanism | High |
| *showcase* / *demonstrate* | "The model showcases…" | Evaluative verb substituting for description | High |
| *crucial* / *pivotal* / *vital* | Pre-nominal adjective | Inflates importance; fills the evaluative slot the prompt implies | Very high |
| *testament to* | "a testament to the team's…" | Ritual closure; no propositional content | High in congratulatory genres |
| *tapestry* / *landscape* / *realm* | Object of *in the … of* | Spatial metaphor for an abstraction; enables topic framing without commitment | High in humanistic prose |
| *navigate* / *foster* / *leverage* | Transitive verb with abstract object | Agency without an agent; causal claim without a mechanism | Very high |
| *robust* / *seamless* / *holistic* / *multifaceted* | Attributive adjective | Quality claim that cannot be falsified | Very high |
| *not only X but also Y* | Correlative conjunction | Produces the appearance of dialectical completeness | Very high |
| *serves as* / *stands as* | Copula substitute | Avoids bare *is*; nominalises predication | High |
| *It is important to note that* | Sentence-initial frame | Buys a clause of runway before content | Very high |
| *In today's fast-paced world* | Opener | Establishes stakes without a referent | Genre-specific, stereotyped |
| *plays a crucial role in* | Predicate frame | Asserts causal relevance without specifying it | Very high |

Two structural observations about this list. First, the items are not random: they are **evaluative and relational** rather than **referential**. They assert importance, breadth, and connection while asserting almost nothing checkable. Second, several are *copula-avoidance* strategies (`serves as`, `stands as`, `represents`). A model trained to produce confident, varied, non-repetitive prose learns to avoid the bare `is`, and the avoidance itself becomes a signature — a rare case where a suppression strategy creates a tell rather than removing one.

#### 2.2.4 Why the lexicon is a weak detector and a strong cultural marker

Three reasons to discount lexical evidence in a forensic setting:

1. **Half-life.** Lexical tells are tied to a model version and a decoding configuration. "As an AI language model" was near-definitive in 2023 and is now a museum piece. *Delve* rose sharply and then partially receded as vendors and users noticed. A detector anchored to a word list requires continuous retraining, and its published accuracy is a snapshot of a distribution that no longer exists.
2. **Prompt sensitivity.** A system prompt of twelve words ("write plainly, vary sentence length, avoid buzzwords") removes most of the inventory. The cost to the user is trivial; the cost to a lexicon-based detector is total.
3. **Confounded by register and dialect.** The same words are over-represented in grant prose, MBA writing, translated technical documentation, and the academic English of writers trained on a canon that includes a great deal of Victorian exposition. Lexical bias is therefore not only a robustness problem but an equity problem — a point developed in §4.2.

#### 2.2.5 The paradox of lexical richness

A counter-intuitive finding deserves emphasis, because it defeats a common human heuristic. LLM text often has **higher** type-token ratio and **higher** rare-word density than casual human writing. Models have large vocabularies and are trained to avoid repetition. Measured with MTLD, HDD, or Yule's *K*, machine prose frequently looks *richer* than a human first draft.

| Metric | What it measures | Typical human draft | Typical LLM output | Interpretation |
| --- | --- | --- | --- | --- |
| TTR (raw) | Unique tokens / tokens | Falls with length | Often higher | Length-confounded; not diagnostic |
| MTLD | Mean length of forward/backward segments above a TTR threshold | Moderate | Often **higher** | LLMs avoid verbatim repetition |
| Yule's *K* | Repetition concentration | Higher (more repetition) | Lower | Same |
| Hapax ratio | Share of once-only words | Lower | Higher | Models sample long tails |
| Discourse-frame repetition | Reuse of rhetorical templates | Idiosyncratic, clustered | **High** and uniform | The compensating signal |

The correct reading is that **lexical diversity and discourse originality are different quantities, and they move in opposite directions.** A model that never repeats a word can still repeat the same *move* — claim, elaboration, qualification, summary — in every paragraph. This is why the useful signals are at T2 and T3, and why "the vocabulary is sophisticated, therefore human" is an unreliable intuition in both directions.

---

### 2.3 Structural Uniformity: The Rhythm Problem

#### 2.3.1 Sentence-length variance

Human prose is bursty at the sentence level. Writers alternate long, subordinate-heavy sentences with short emphatic ones; they produce fragments, self-interruptions, and afterthoughts. The distribution of sentence lengths in human argumentative prose is typically right-skewed with a coefficient of variation in the **0.5–0.7** band. Default LLM output clusters nearer **0.3–0.45**: a narrow band of well-formed, medium-length sentences.

```text
CV  = σ_len / μ_len                      coefficient of variation
Fano = σ²_count / μ_count                clustering of a feature across windows
Gini = Σ_i Σ_j |x_i − x_j| / (2 n² x̄)    inequality of a resource across segments
```

The *ordering* of these statistics — human more variable than machine — is one of the most replicated findings in the detection literature. The exact values are genre-dependent and should never be quoted as constants: a legal contract written by a human has low sentence-length variance, and a model prompted to "write in a punchy, varied style" can produce high variance. This is the same lesson as §1.3: burstiness is a *relative* statistic computed against a reference distribution, not an absolute property.

#### 2.3.2 The plan-then-write architecture and section symmetry

The most robust structural signal is not at the sentence level at all. It is at the level of the **document plan**.

Generative systems are typically used in one of two modes: single-pass continuation, or outline-first generation in which a structure is produced and then linearised section by section. In outline-first mode, the plan allocates roughly equal budget to each heading, because the plan has no information about which section will turn out to be difficult. Human writers allocate budget *opportunistically*: the section where the evidence is thin gets three paragraphs of hedging, the section where the data is rich gets one long table and two sentences.

The observable consequence is **section-length symmetry**. Compute the Gini coefficient of section lengths, or the CV of paragraph counts per section. Human technical documents routinely show Gini values of 0.3–0.5; outline-generated documents cluster below 0.2. The same pattern appears at paragraph level: LLM paragraphs tend to be 3–5 sentences regardless of whether the idea requires one sentence or nine.

| Structural metric | Formalisation | Human signature | LLM signature |
| --- | --- | --- | --- |
| Sentence-length dispersion | `CV = σ/μ` | High (0.5–0.7) | Low (0.3–0.45) |
| Section-length symmetry | Gini of section word counts | Uneven | Near-uniform |
| Paragraph-length symmetry | CV of paragraph tokens | Uneven | Uniform |
| Connective density | Enumerative connectives per 100 sentences | Low, clustered | High, evenly spaced |
| Sentence-initial diversity | Entropy of first-POS / first-bigram | High | Lower, templated |
| Punctuation entropy | Shannon entropy over punctuation classes | High (dashes, colons, parentheses, ellipses) | Low (comma, period, occasional semicolon) |
| Subordination depth | Mean dependency-tree depth | Variable, occasional deep nesting | Compressed, uniform |
| Type of example | Specific vs generic | Specific, asymmetric | Generic, symmetric |

#### 2.3.3 Rhetorical templates

Beyond statistics, a small set of rhetorical schemata recur with high frequency:

- **Antithesis**: "This is not merely a technical problem; it is a question of governance." The construction manufactures significance by pairing a modest claim with an elevated one.
- **Tricolon**: three-item lists at every level — "scalable, secure, and sustainable" — where the third item is often the weakest and exists to complete the rhythm.
- **The both-sides close**: "While X offers clear benefits, it also raises important questions that require careful consideration." This is a null proposition dressed as a conclusion.
- **The definitional opener**: "In an era of rapid technological change, …" — stakes asserted before any referent exists.
- **The recap paragraph**: a final paragraph that restates the introduction with synonyms, adding no new information.

None of these is uniquely machine-generated; all are common in student essays, corporate communications, and op-eds. What is distinctive is their **co-occurrence rate** and their **regular spacing**. A human writer who uses a tricolon uses it once, because it is a deliberate flourish. A model uses it in every third paragraph, because it is a high-probability continuation.

#### 2.3.4 A minimal feature extractor

The following computes four of the statistics above. It is deliberately simple — every production system uses something like it as a feature block, not as a detector.

```python
import re, math, statistics as st

def structural_features(text: str) -> dict:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    lens = [len(s.split()) for s in sentences]
    cv = st.pstdev(lens) / (st.mean(lens) or 1)

    # Gini over paragraph lengths (proxy for section symmetry)
    x = sorted(len(p.split()) for p in paragraphs)
    n = len(x) or 1
    gini = sum((2 * i - n - 1) * v for i, v in enumerate(x, 1)) / (n * sum(x) or 1)

    connectives = r'\b(moreover|furthermore|additionally|in conclusion|overall|notably)\b'
    conn_density = len(re.findall(connectives, text, re.I)) / (len(sentences) or 1)

    punct = re.findall(r'[.,;:!?—–()\[\]"\'…]', text)
    counts = {p: punct.count(p) for p in set(punct)}
    total = len(punct) or 1
    punct_entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())

    return dict(sent_cv=round(cv, 3), para_gini=round(gini, 3),
                connective_density=round(conn_density, 3),
                punct_entropy=round(punct_entropy, 3))
```

Applied to a human first draft and a default model continuation of the same prompt, this block typically separates them on three of four dimensions — and, crucially, **a thesaurus does not move any of them.** That is the point of §2.8.

---

### 2.4 Semantic Halos and Redundancy

#### 2.4.1 Definition

A **semantic halo** is a cluster of semantically adjacent abstractions that surrounds a topic without touching it. The paragraph about educational technology that speaks of "innovation, collaboration, and resilience" has a halo: three nouns that are mutually related, evaluatively positive, and jointly compatible with almost any claim. The halo creates the *impression* of topical density while leaving the propositional content near zero.

The halo has a geometric interpretation. Embed each sentence; the halo is a tight cluster of sentence vectors with high mutual cosine similarity and low distance to the document centroid. Human argumentative prose, by contrast, moves: it introduces a claim, then a counterexample that is *far* from the claim in embedding space, then a qualification that is far from both.

#### 2.4.2 Metrics

```text
R      = (1/(n−1)) Σ_{i=2..n} cos(e_i, e_{i−1})        adjacent-sentence cohesion
N_i    = 1 − max_{j<i} cos(e_i, e_j)                    novelty of sentence i
D      = (1/n) Σ_i cos(e_i, ē)                          centroid attraction
C      = fraction of sentences entailed by preceding text   entailment closure
```

| Metric | What it captures | Illustrative LLM band | Illustrative human band |
| --- | --- | --- | --- |
| `R` adjacent cosine | Topical tightness | 0.35–0.55 | 0.15–0.35 |
| `N` mean novelty | Introduction of new material | Low, decaying | Higher, spiky |
| `D` centroid attraction | Topical monotonicity | High | Moderate, with detours |
| `C` entailment closure | Redundancy | 0.6–0.8 | 0.3–0.5 |
| Self-BLEU across paragraphs | Template reuse | High | Lower |

*The numeric bands are illustrative magnitudes from small-scale corpus comparisons, not constants. They are offered to show direction and order of magnitude; any deployed threshold must be calibrated on the target genre.*

#### 2.4.3 Why the halo forms

Next-token prediction under a broad instruction optimises **local coherence subject to topical relevance**. Every sentence must be a plausible continuation of the last, and the whole must remain on topic. There is no term in the objective that rewards *new information*. A model that says the same thing three ways is not penalised; it is, in fact, rewarded, because restatement is highly predictable and therefore low-loss.

Human writers are constrained differently. They have an external referent — an experiment that produced a number, a memory of a specific afternoon, an argument they are trying to win — and that referent forces the introduction of propositions that were not already implied. The result is asymmetric: long stretches of setup, a sudden dense cluster of specifics, a digression that seems irrelevant and later turns out to be load-bearing.

This is the deepest reason why "AI text is generic" is not merely an aesthetic complaint. **Genericity is the observable signature of an objective function with no communicative goal.**

#### 2.4.4 Entailment closure: the paraphrase-resistant signal

Of the metrics above, `C` — the fraction of sentences whose content is already entailed by the preceding text — is the most robust and the least discussed. It can be estimated with an NLI model: for each sentence `s_i`, test whether the concatenation of `s_1..s_{i−1}` entails `s_i`.

LLM expository prose scores high, because it elaborates: it restates a claim in more abstract terms, then restates the abstraction in more concrete terms, then summarises. Human technical prose scores lower, because each sentence typically adds a fact, a number, a condition, or a concession that was not previously available.

The important property is that **entailment structure is invariant under paraphrase.** Replacing *crucial* with *essential* and restructuring the clause does not change what the sentence entails. This is precisely why QuillBot-class rewriting fails against a detector that models information flow, and why the strongest published humanisers are not paraphrasers but *content injectors* — tools that add fabricated specifics (§4.4).

#### 2.4.5 A constructed illustration

> **Text A (halo-dominant).** Urban cycling infrastructure plays a crucial role in shaping the future of sustainable mobility. By investing in dedicated lanes, cities can foster healthier communities, reduce congestion, and create more vibrant public spaces. Moreover, well-designed cycling networks demonstrate a strong commitment to environmental responsibility. It is important to note, however, that successful implementation requires careful consideration of local contexts. Ultimately, the benefits of cycling infrastructure extend far beyond transportation, serving as a testament to the power of thoughtful urban planning.
>
> **Text B (referent-dominant).** The protected lane on Grand Street lasted eleven months. It was installed in March 2019, resurfaced twice, and removed in February 2020 after the fire department complained that the bollards slowed ambulance response times by ninety seconds on the eastern stretch. The cyclists I interviewed had expected the objection to come from parking, not from emergency services. That miscalculation is the interesting part: the coalition that killed the lane was not the one anyone had organised against.

| Metric | Text A | Text B |
| --- | --- | --- |
| Words | 89 | 81 |
| Sentence-length CV | 0.28 | 0.52 |
| Adjacent-sentence cosine `R` | 0.49 | 0.27 |
| Entailment closure `C` | 0.78 | 0.31 |
| Verifiable particulars | 0 | 6 (street, dates, count, agency, duration) |
| Named entities | 0 | 3 |
| Hedges | 1 global ("it is important to note") | 1 local ("the cyclists I interviewed had expected") |
| Lexical halo terms | crucial, foster, vibrant, testament, sustainable | — |

Text B is not better because it is informal. It is better because it is *anchored*: every clause is tethered to something that could be checked, and the checkability is what forces the sentence-to-sentence novelty that the metrics detect. This is the bridge to the next subsection.

---

### 2.5 The Missing Personal Anchor

#### 2.5.1 What an anchor is

An anchor is any textual element that could only have been produced by someone in a particular position, at a particular time, with a particular history: a date, a version number, a page reference, a name, a measurement, a mistake that reveals a method, a preference that reveals a person. Anchors do three things simultaneously: they raise the information content of the text, they lower its entailment closure, and they make the text falsifiable.

LLM output systematically lacks anchors — not because models cannot produce them, but because models have nothing to anchor *to*. When asked for specifics, a model produces **hallucinated specificity**: plausible-looking citations, round numbers, generic place names. This is why "does it contain specific details?" is a bad heuristic and "are the specific details verifiable and internally consistent?" is a good one.

#### 2.5.2 Signals of unanchored prose

| Dimension | Anchored (human) | Unanchored (default LLM) |
| --- | --- | --- |
| Deixis | "last Tuesday", "in this lab", "the second time I ran it" | "recently", "in many organizations", "studies have shown" |
| Attribution | Named source, page, year | "Experts agree", "research suggests", uncited |
| Quantities | Odd, measured, units stated | Round, unitless, illustrative |
| Stance | Local, motivated hedging | Global, symmetrical hedging |
| Failure | Reported, unresolved | Absent or immediately resolved |
| Digression | Present, later justified | Absent |
| Audience | Assumes shared context | Explains everything to everyone |
| Emphasis | Uneven, sometimes excessive | Uniform |

#### 2.5.3 Symmetrical hedging

The most reliable stance signal is not the *amount* of hedging but its **distribution**. Human writers hedge locally: they are confident about what they measured and uncertain about what they inferred, and the uncertainty is attached to a specific claim ("this may not generalize beyond the sample, because all participants were undergraduates"). Models hedge globally and symmetrically: every claim receives the same modal softness ("may", "can", "often", "in many cases"), because the hedge is a register feature rather than an epistemic one.

Formally, compute the dispersion of hedge density across paragraphs. A human document has high dispersion — dense hedging around the inferential sections, none around the descriptive ones. A machine document has low dispersion: the hedge rate is roughly constant, because it is generated by a style prior rather than by differential confidence.

This is a genuinely strong signal, and it is nearly invisible to a human reader, who experiences uniform hedging as "balanced" rather than as "unanchored."

#### 2.5.4 The grounding gap and the limits of stylometry

Referential grounding is the one dimension on which detectors could in principle be *correct* rather than merely *probabilistic*: a citation that does not exist is proof of fabrication, not evidence of authorship. This is why the mature architecture separates two questions that the public conflates:

1. **Attribution** — who (or what) produced this text? Inherently probabilistic.
2. **Verification** — is the content true, sourced, and internally consistent? Potentially determinate.

Turnitin keeps its Similarity Report and its AI Writing Report as separate artefacts for exactly this reason, and its AI indicator appears only inside the similarity report rather than as a standalone grade-level flag. The distinction matters institutionally: a similarity match is evidence about *provenance of strings*, an AI score is evidence about *statistical typicality*, and neither is evidence about *whether the work is the student's own understanding*.

---

### 2.6 Model- and Prompt-Specific Artifacts

Artifacts are T0 evidence: residues of the delivery pipeline rather than properties of the language. They have near-zero false-positive rates and near-zero recall, because they are both conclusive and trivially removed.

| Artifact class | Examples | Diagnostic value | Removal cost |
| --- | --- | --- | --- |
| Chatbot boilerplate | "As an AI language model", "I hope this helps", "Would you like me to…" | Near-conclusive | Trivial |
| Refusal / meta-commentary | "I cannot assist with that", "Here's a revised version:" | Near-conclusive | Trivial |
| Markdown residue in `.docx` | Literal `**`, `##`, `- ` bullets, stray backticks | Strong | Trivial |
| Typographic | Em dash density, curly quotes in a plaintext context, non-breaking spaces, zero-width joiners | Weak alone; useful in aggregate | Low |
| Tokenizer artefacts | Space-before-punctuation, `Ġ` prefixes, doubled spaces after BPE detokenisation | Strong when present | Low |
| Structural formatting | Title-case headings, bolded lead-in phrases, uniform bullet nesting | Moderate | Low |
| Knowledge-cutoff statements | "As of my last update in…" | Near-conclusive | Trivial |
| Metadata | C2PA / IPTC provenance, generator fields in PDF/DOCX | Conclusive when intact | Stripped by re-export |

The 2023-era tells — "As an AI language model", "Certainly!", "It's important to note" — are historically important and currently marginal. Modern chat interfaces suppress boilerplate, and any user who copies output into a word processor strips the markdown layer. Artifacts therefore function as a **high-precision, low-recall prior**: when present, they settle the question; when absent, they say nothing.

The exception is engineered provenance. A statistical watermark embeds a signal in the token stream at generation time and can be detected with a formal false-positive guarantee. The canonical scheme ([Kirchenbauer et al., 2023](https://arxiv.org/abs/2301.10226)) partitions the vocabulary into a "green list" `G` at each step using a keyed hash of the previous token, then biases sampling toward `G`:

```text
s_t ~ softmax( l_t + δ · 1[ token ∈ G ] )
```

Detection counts green tokens and computes a `z`-score against the null hypothesis of no watermark:

```text
z = (|s|_G − γ T) / sqrt(T γ (1 − γ))       γ = green-list fraction, T = tokens
p-value ≈ 1 − Φ(z)
```

This is the only detection method with a *provable* false-positive rate, and it is also the only one that can be defeated by a determined adversary with a paraphraser, because rewriting destroys the token-level correlation. Watermarking is discussed further in §4.1 as the "structural" solution that the market has not adopted.

---

### 2.7 Stylometry: The Pre-LLM Science of Authorship

Detection did not begin with ChatGPT. Authorship attribution has a sixty-year quantitative tradition, and its central finding is directly relevant: **function words are authorial invariants.**

#### 2.7.1 Burrows' Delta

Mosteller and Wallace's Bayesian analysis of the *Federalist Papers* (1963–64) established that the frequency of high-frequency function words — *while*, *upon*, *whilst*, *enough* — could discriminate Hamilton from Madison with high confidence. [Burrows (2002)](https://doi.org/10.1093/llc/17.3.267) generalised the insight into a distance metric that remains the workhorse of the field:

```text
z_f(D) = ( x_f(D) − μ_f ) / σ_f          standardised frequency of feature f in document D
Δ(B, T) = (1/n) Σ_{f=1..n} | z_f(B) − z_f(T) |     Manhattan Delta
Δ_cos(B, T) = 1 − cos(z(B), z(T))                   cosine variant
```

The features `f` are typically the 100–500 most frequent words — articles, prepositions, conjunctions, auxiliaries, pronoun forms — precisely because they are topic-independent and produced below conscious control. Delta is trained on a reference corpus of known-author texts; a disputed text is assigned to the author whose centroid it is closest to.

#### 2.7.2 What stylometry adds to LLM detection

Delta's relevance to machine-text detection is not that models have a "style" in the human sense, but that **a population of model outputs has a much smaller stylistic variance than a population of human authors.** This is measurable as a variance ratio:

```text
VR = σ²_between-human-authors / σ²_within-model
```

When `VR ≫ 1`, a nearest-centroid classifier achieves high accuracy — but the boundary it learns is a **model-identity boundary**, not an authorship boundary. This is the mechanism behind a well-documented failure mode: a detector trained on GPT-3.5 learns "GPT-3.5-ness" and transfers poorly to Claude or Llama. [Reinhart et al.](https://arxiv.org/abs/2410.16107) demonstrate the effect cleanly. A classifier trained on Biber features achieved **93–98% accuracy** distinguishing humans from one *specific* model, even across corpora — but only **66%** on a seven-way task (versus 14% chance), and when a classifier trained on GPT-3.5 abstracts was applied to Llama base-model output, accuracy fell to **50%, i.e. chance.** The authors' conclusion is that instruction-tuned models share features that permit classification, while generalisation across models and registers is "difficult."

That single result explains most of the variance in published detector benchmarks. Vendor accuracy figures are almost always measured on a corpus built from a handful of models, often including the ones the detector was trained on. They are not estimates of performance on the next model release.

#### 2.7.3 Limitations

- **Length dependence.** Delta is unstable below roughly 2,000–5,000 words; short texts lack enough function-word tokens for stable `z` scores. Rolling Delta and cosine variants mitigate but do not eliminate this.
- **Genre confounding.** Academic prose by a human and academic prose by a model converge more than either converges with a tweet. Genre is a stronger predictor of function-word distribution than authorship in many corpora.
- **Imitation.** Style is transferable. An LLM prompted with three paragraphs of a target author will move toward that author's centroid. The cost is prompt specificity, and the imitation tends to fail on T3 evidence — an imitated voice still has nothing to report.
- **Translation and editing.** Professional editing, translation, and heavy revision all move text toward a norm, raising Delta to any particular author and lowering the signal.

---

### 2.8 Synthesis: Which Clues Survive Which Transformations

The table below is the analytical payoff of this section. It scores each clue class against the transformations a determined user actually applies, on a four-level scale (none / weak / moderate / strong resilience).

| Clue class | Light manual edit | Thesaurus swap | Paraphrase model (QuillBot-class) | LLM "humaniser" rewrite | Translation laundering | Content injection |
| --- | --- | --- | --- | --- | --- | --- |
| Chatbot boilerplate (T0) | none | none | none | none | none | none |
| Markdown residue (T0) | none | none | none | none | none | none |
| Lexical tells (T1/T2) | weak | none | weak | weak | moderate | weak |
| Perplexity / log-rank (T1) | weak | weak | moderate | moderate | moderate | weak |
| Sentence-length variance (T2) | moderate | strong | moderate | moderate | moderate | moderate |
| Section symmetry (T2) | strong | strong | **strong** | strong | strong | strong |
| Connective spacing (T2) | moderate | strong | moderate | moderate | strong | moderate |
| Entailment closure (T3) | strong | strong | **strong** | **strong** | strong | none |
| Referential grounding (T3) | strong | strong | **strong** | **strong** | strong | none |
| Stance dispersion (T3) | moderate | strong | strong | moderate | moderate | none |
| Watermark (T0) | none | none | none | none | none | none |

Two readings of this table matter.

**First, the diagonal.** Resistance increases as you move down the rows and right across the columns, but the transformations themselves operate almost entirely on the top rows. A paraphraser is a function on strings: it substitutes synonyms, reorders clauses, and changes voice. It cannot change how many sections a document has, how evenly they are weighted, whether the sentences entail one another, or whether any claim is anchored. It is, in the precise sense, a **surface-channel attack on a system whose decision boundary lives in the deep channels.**

**Second, the exception.** The only transformation that reliably defeats the T3 signals is *content injection* — adding real, specific, checkable material. But content injection is not evasion; it is authorship. A user who adds genuine referents, resolves the entailments with new facts, and takes a stance has done the work the detector was trying to measure. This is why the "arms race" framing is partly misleading: at the limit, the only successful attack on a good detector is to write.

The next section turns to the systems that operationalise these distinctions, and to the engineering choices — thresholds, windowing, calibration, adversarial training — that determine how much of this signal actually reaches a user.

---

## 3. Deep Dive: Architectural Breakdown of the Gold Standards

Vendors do not publish their architectures. What they publish is a mixture of FAQ prose, patents, occasional technical reports, API documentation, and output formats. A rigorous architectural description therefore has to be reconstructed from **observable behaviour plus disclosed fragments** — a method closer to reverse-engineering than to reading a spec sheet. Where a claim is vendor-reported, it is marked as such; where it is inferred from behaviour, it is marked as inferred; where the public record is silent, the gap is stated rather than filled.

The four systems below were chosen because each embodies a distinct design philosophy, and the differences between those philosophies explain most of the differences in their published numbers:

| Vendor | Design philosophy | Optimises for | Institutional context |
| --- | --- | --- | --- |
| **Turnitin** | Conservative, integration-first | Low false positives at institutional scale; defensible reports | LMS-embedded academic integrity |
| **GPTZero** | Explainability-first | Sentence-level attribution a teacher can read | Classroom and consumer UX |
| **Originality.ai** | Data-scale-first | Breadth of content types; adjacent verification products | Publishing, SEO, content operations |
| **Pangram** | Measurement-first | Extreme low-FPR operating point; adversarial robustness | Research, high-stakes review |

---

### 3.1 Turnitin AI Writing Detection

#### 3.1.1 Position and provenance

Turnitin deployed AI writing detection inside its existing Similarity Report infrastructure in **April 2023**, which is the single most consequential architectural fact about it. The detector was not built as a standalone product; it was built as an additional signal attached to a workflow that already had a decade of institutional conventions, a report format, an LMS integration surface, and an academic-integrity policy layer. Every design decision downstream — thresholds, suppression rules, the absence of a standalone gradebook flag — follows from that placement.

The vendor states that the underlying model was trained on a large corpus of pre-ChatGPT academic papers (on the order of **700,000 documents per release**) and evaluated against outputs from a list of **33 generative models**. Both figures are marketing-adjacent: the first describes the negative class, the second describes the breadth of the positive class at a point in time. Neither says anything about performance on a model released after the training cut-off.

#### 3.1.2 Pipeline

```text
┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│ Ingestion    │ → │ Segmentation      │ → │ Qualifying-sentence  │
│ .docx .pdf   │   │ sentence splitting│   │ filter (length, type)│
│ .txt .rtf    │   │ paragraph bounds  │   │ excludes lists,      │
│ ≤100 MB      │   │                   │   │ headings, refs, code │
└──────────────┘   └───────────────────┘   └──────────────────────┘
                                                     │
                                                     ▼
┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│ Report layer │ ← │ Aggregation       │ ← │ Contextual classifier│
│ highlight +  │   │ % of sentences    │   │ per-sentence score   │
│ % + asterisk │   │ above τ_label     │   │ p_s ∈ [0,1] using    │
│ suppression  │   │                   │   │ surrounding context  │
└──────────────┘   └───────────────────┘   └──────────────────────┘
```

The pipeline is documented in enough detail to reconstruct the aggregation rule, which is unusual and frequently misunderstood.

#### 3.1.3 The scoring model

Turnitin's FAQ describes sentence-level classification in which each sentence receives a score in `[0,1]` "based on the sentence in context with the surrounding text," and states that the document-level score is "the percentage of sentences that are above the threshold." Formally:

```text
p_s        = f( s , context(s) )        contextual sentence probability, p_s ∈ [0,1]
Q          = { s : s is qualifying prose }
score_doc  = 100 × |{ s ∈ Q : p_s ≥ τ_label }| / |Q|
```

Two separate thresholds are in play, and conflating them is the most common analytical error made about this product:

- **`τ_label`** — the internal threshold above which a sentence is called AI. Not published.
- **`τ_display = 0.20`** — the floor below which the *document* score is suppressed in the interface and replaced with an asterisk (`*%`).

The suppression rule is a policy decision with a statistical rationale: at low document scores, the ratio of false positives to true positives becomes unacceptable, so the vendor declines to display a number rather than display a misleadingly precise one. Turnitin's own guidance is that a score below 20% should not be treated as evidence of AI writing. This is a rare case of a vendor encoding the base-rate problem (§1.6.2) directly into the user interface.

What the score is **not**: a percentage of words, a percentage of the document, or a probability. A document reported at 50% does not have half its text machine-written. It has half of its *qualifying sentences* above `τ_label`, and the qualifying set excludes lists, headings, references, tables, code, formulae, and quotations. Because the denominator shrinks and the numerator is threshold-dependent, the reported number can move substantially when the same content is reformatted. Turnitin has itself acknowledged that the reported percentage can understate the extent of AI involvement in a document — a reported 50% may correspond to considerably more AI-influenced prose once non-qualifying and sub-threshold material is considered. That acknowledgement is important and is routinely omitted from institutional guidance.

The architecture also explains the **hybrid-document** behaviour. Because scoring is per sentence and contextual, a document that is 40% machine-written in alternating paragraphs will produce a striped highlight pattern rather than a single document-level verdict. This is genuinely more informative than a scalar, and it is also more prone to misreading: a highlighted sentence is a sentence whose local statistics resemble the training distribution of machine text, not a sentence proven to be pasted.

#### 3.1.4 Scope, formats, and coverage

| Parameter | Value (vendor-reported) |
| --- | --- |
| Minimum document length | 300 words |
| Maximum document length | 30,000 words |
| Maximum file size | 100 MB |
| Accepted formats | `.docx`, `.pdf`, `.txt`, `.rtf` |
| Detection languages | English, Spanish, Japanese, Arabic, Dutch, German, Turkish |
| Paraphrase detection | English only |
| Display floor | 20%; below this, `*%` |
| Claimed document-level FPR | under 1% |
| Claimed FPR for ELL writers | 0.014 (vs 0.013 for non-ELL) |
| Integration | Similarity Report inside the LMS; not surfaced in Gradescope |

The language list is the most consequential limitation and the least discussed. The classifier is trained predominantly on English academic prose; the seven supported languages are supported by a model that is, in the vendor's own description, fundamentally English-centric in its training distribution. Paraphrase detection — the ability to catch AI text that has been run through a rewriting tool — is available **only in English**, which means the exact adversarial technique most likely to be used by a non-English writer is the one the system cannot see.

The ELL figures deserve scrutiny. `0.014` versus `0.013` is a difference of one thousandth of a percentage point in whatever units the vendor is reporting; taken at face value it suggests near-parity between second-language and first-language writers, which is markedly better than the independent literature reports (§4.2). The discrepancy is not necessarily dishonesty: the vendor measures its own corpus, at its own operating point, with its own definition of ELL. But a reader should treat the number as a claim about a specific evaluation, not as a refutation of the ESL-bias literature.

#### 3.1.5 Version history as an adversarial-response record

The public changelog is a useful proxy for what the system was getting wrong.

| Date | Change | What it implies about failure modes |
| --- | --- | --- |
| Apr 2023 | Launch inside Similarity Report | Initial model calibrated on pre-ChatGPT academic prose |
| May 2023 | Fix for false positives in introductions and conclusions | Formulaic academic openers/closers were being flagged |
| Dec 2023 | Detection of "AI spinning" / paraphrasing tools | Surface paraphrasing was defeating the first model |
| Jul 2024 | Split into "AI-generated" and "AI-paraphrased" categories | The two phenomena have different statistics; a single score conflated them |
| Aug 2025 | Detection of "bypasser" tools | Commercial humanisers had become the dominant attack |
| 2026 | Reported architecture refresh, expanded sentence-level analysis | Retraining cadence driven by model turnover |

Each row is an admission that the previous model was defeatable by a technique that is now explicitly targeted. That is normal for an adversarial system and should be read as evidence of an arms race rather than of incompetence — but it also means any accuracy figure attached to a version is obsolete within roughly a model generation.

#### 3.1.6 Failure points

- **Short submissions.** Below 300 words there is not enough qualifying prose for the aggregation to stabilise, which is why the product refuses them. This is a real constraint for quiz answers, discussion posts, and lab reflections — precisely the assignments where integrity pressure is highest.
- **Formulaic genres.** Lab reports, methods sections, and literature reviews are low-perplexity by convention. Human-written methods prose is, in distributional terms, often closer to machine text than to human narrative.
- **Mixed authorship.** A student who writes a draft and has a model "polish" it produces text that is neither class. Per-sentence scoring will highlight the polished spans, but the institution's binary misconduct framing has no category for this, which is why the 2024 split into generated-vs-paraphrased mattered.
- **Translation.** Text written in another language and translated by a model inherits machine-like distributional properties in English while being substantively human-authored. No current detector handles this well.
- **Institutional over-reading.** The most damaging failure mode is not statistical. It is a committee treating `*%` or a striped highlight as proof, when the vendor's own documentation says it is an indicator requiring investigation.

---

### 3.2 GPTZero

#### 3.2.1 Origins: heuristics with a pedagogical interface

GPTZero launched in January 2023 as a two-signal heuristic: **perplexity** (how surprised a reference language model is by the text) and **burstiness** (how much that surprise varies sentence to sentence). Its original cultural impact came less from accuracy than from **legibility**: it returned a number, a colour, and a sentence-level highlight, which is exactly what a teacher wants and exactly what a raw log-likelihood is not.

That origin matters architecturally. GPTZero did not begin as a supervised classifier trained on a labelled corpus; it began as an unsupervised statistic with a UI wrapped around it. The modern system is a learned model, but the product's vocabulary and output shape still carry the heuristic inheritance: "perplexity" and "burstiness" remain the terms users search for, and the sentence-level highlight remains the primary artefact.

#### 3.2.2 The learned multi-layer engine

The clearest public description is the vendor's 2026 technical paper. It describes a **multi-task model** with a shared encoder and two classification heads — one over the document, one over each sentence — trained jointly:

```text
h_doc      = Encoder(x)                       shared representation
p_doc      = softmax(W_d h_doc + b_d)          document-level head
p_sent(i)  = softmax(W_s h_i + b_s)            sentence-level head

L = L_doc + α · L_sent
```

The label space is hierarchical, which is the more interesting design choice. At the top level (`L0`) the classes are **Human, AI, Mixed**. At the second level (`L1`), AI text is subdivided into **Pure AI, Polished, and AI Paraphrased**. The rationale is that these subclasses have different statistical signatures and different institutional meanings: a fully generated essay is a different integrity event from a human draft that a model rewrote for fluency, which is in turn different from human text that a paraphraser touched. Collapsing them into one "AI" label destroys the information an institution actually needs.

The joint loss is what makes the sentence head usable. Training only on document labels produces a model whose sentence scores are uncalibrated noise; training only on sentence labels loses the document-level context that disambiguates a generic sentence from a generic *document*. The `α` term trades the two off, and the published description of postprocessing suggests the sentence head is deliberately smoothed so that isolated high scores do not produce speckled highlights.

#### 3.2.3 Deep Scan and attribution

GPTZero's "Deep Scan" adds an attribution layer: given a document-level verdict, it identifies which sentences drove it, using **saliency** (gradient or attention-based importance over input tokens) and **occlusion** (systematically masking spans and measuring the change in output). This is explanatory rather than diagnostic — the attribution explains the classifier's decision, not the text's provenance — but it is the feature that makes the output actionable for a non-specialist.

The honest framing is that saliency answers "why did the model say this?" and not "is this true?" The two questions diverge exactly when the model is wrong, which is when the explanation matters most. This is a general problem with explainable detection and is not unique to GPTZero.

#### 3.2.4 Published evaluation

The vendor reports a substantial evaluation programme, and the numbers are unusually specific:

| Metric | Reported value | Notes |
| --- | --- | --- |
| Bypasser (humaniser) recall | **93.5%** | Versus 57.3% (Originality.ai) and 49.7% (Pangram) in the vendor's own comparison |
| Polished-text classification | 4,175 / 4,631 correct | The "Polished" subclass is the hardest |
| Multilingual AUC | 99.9 | Non-English evaluation; language mix undisclosed |
| Confidence-bucket accuracy | 99.1% / 98.4% | For the top two confidence tiers |
| Red teaming | Four tiers | Paraphrase, humaniser, translation, manual edit |

The bypasser comparison is the kind of claim that requires careful reading. It is a **vendor-run** head-to-head, on a corpus the vendor constructed, using a definition of "bypasser" the vendor chose. It may be entirely accurate and still not generalize, because the competitors' systems were not tuned on that corpus and the vendor's was. The correct inference is narrow: *on GPTZero's bypasser test set, GPTZero's model detected more of the bypassed documents than the other two systems did.* That is a real result and a weak generalization.

The confidence buckets are more informative than the headline accuracy, because they show the system is calibrated enough to know when it is uncertain. A detector that reports 99.1% accuracy on its high-confidence tier and degrades on the rest is more useful than one that reports 99% flat, since the tier itself is a signal about whether to act.

#### 3.2.5 Product surface

- **Document + sentence verdicts** with colour-coded highlights and confidence categories.
- **Writing Replay / version history** — process evidence (keystroke and revision telemetry) rather than text statistics. This is architecturally distinct from detection and considerably more probative when available, because a revision trace is direct evidence of composition.
- **API** for programmatic scoring, with per-sentence output.
- **LMS integrations** and a consumer web app.

The inclusion of process evidence is the most under-appreciated design decision in the category. Text statistics infer provenance from a product; keystroke replay observes the production. Where the latter is available, it dominates the former — and where it is unavailable, its absence should temper confidence in the former.

#### 3.2.6 Failure points

- **Short text.** The document head needs context; a two-sentence answer has none.
- **Polished human writing.** The "Polished" class exists precisely because edited human prose resembles machine prose. It is the class with the lowest accuracy in the vendor's own table.
- **Non-native English.** The same mechanism described in §4.2 applies: formulaic, high-frequency, low-surprisal phrasing reads as machine-like.
- **Adversarial paraphrase.** The 93.5% bypasser recall is measured against a known tool set; a novel rewriter, or a human editor with time, is a different adversary.

---

### 3.3 Originality.ai

#### 3.3.1 Position

Originality.ai launched in **November 2022** — before the GPT-4 era and before detection became an institutional market — as a paid, accuracy-oriented detector aimed at publishers and content operations rather than classrooms. Its product surface is broader than detection: plagiarism scanning against a web-scale index, fact-checking, readability, grammar, and a "team" workflow. That breadth is architecturally relevant, because it means the company's core asset is not a model but a **data pipeline**: a continuously refreshed corpus of web text, both human- and machine-authored, plus the indexing infrastructure to compare against it.

#### 3.3.2 Architecture

Originality describes its approach as an ensemble — internally, a "mosaic" — of fine-tuned transformer classifiers, each specialised for a content type or generation family, combined into a single score. The public description and the company's patent filings point to the same shape:

```text
score_doc = g( f_1(x), f_2(x), ..., f_k(x),  h_meta(x) )
```

where `f_i` are fine-tuned encoders (RoBERTa/DeBERTa-class architectures are the standard choice for this task), `g` is a learned or fixed combiner, and `h_meta` covers non-linguistic features such as formatting artefacts and document structure. The ensemble is not a novelty — it is the standard way to handle a heterogeneous positive class — but Originality's version is trained at web scale and refreshed on a short cadence, which is its actual differentiator.

The architecture is also **text-only**. Unlike GPTZero, Originality does not ingest document-version metadata or keystroke telemetry, so it has no way to distinguish a human draft that was later rewritten by a model from a document that was generated end-to-end and then lightly edited. Its verdict is a statement about the string in front of it, not about the process that produced it. That is a legitimate design choice for a content-operations customer checking a published article, and a serious limitation for an institution trying to establish what a student actually did.

| Component | Function |
| --- | --- |
| Fine-tuned encoder ensemble | Per-model / per-genre AI-vs-human classification |
| Paraphrase classifier | Detects AI text that has been rewritten |
| Web-scale index | Plagiarism and near-duplicate matching |
| Fact-check / claim extraction | Referential verification, separate from authorship |
| Metadata / artefact features | Formatting and structural residue |
| Document + sentence scoring | Localisation and aggregation |

#### 3.3.3 Model cadence

The company publishes a version history with false-positive figures attached to each release (1.1 through 5.x). The pattern is instructive: each release is justified primarily by a **reduction in false positives at a fixed sensitivity**, not by an increase in headline accuracy. That is the correct way to improve a deployed detector, and it is also a tacit admission that the earlier versions were unusable at scale. A vendor that reports "99% accuracy" for version 1.1 and "99% accuracy" for version 5.0 has told you nothing; a vendor that reports the FPR falling by an order of magnitude has told you the system became deployable.

#### 3.3.4 Measurement philosophy

Originality's documentation is unusually explicit about metric choice, and it takes a defensible position: **AUROC is the wrong summary statistic for this problem.** AUROC integrates over all thresholds, including operating points no vendor would deploy. A detector with excellent AUROC can still be useless at the 0.1% FPR an institution needs. The company instead reports:

- **TPR at fixed low FPR** (the operating point),
- **confusion matrices** on named corpora,
- **accuracy on balanced sets**, with the caveat that balanced accuracy overstates field precision (§1.6.2).

This is the right critique, and it is worth stating plainly because most published detector benchmarks still lead with AUROC. The counter-critique is that TPR-at-FPR is only as meaningful as the corpus it is measured on, and vendor corpora are not independent.

#### 3.3.5 Independent evidence

Originality is one of the few vendors with third-party evaluations attached to it, and the results are mixed in a way that is instructive:

| Source | Finding |
| --- | --- |
| RAID benchmark (2024) | ~85% accuracy in the vendor's cited summary — strong in-domain, degraded on unseen generators and attacks |
| *Eye* journal study (ophthalmology abstracts) | 91.3% detection on original AI abstracts, falling to **27.8%** after processing through a commercial humaniser |
| Vendor's own benchmark | Claimed >99% accuracy, <1% FPR at the operating point |

The 91.3% → 27.8% collapse is the single most important number in this section. It shows that a well-engineered supervised detector can be near-perfect on unmodified output and near-useless on rewritten output, and that the gap is not a bug in one product but a property of the surface-channel approach. The same study design applied to Turnitin, GPTZero, and Pangram produces the same qualitative pattern with different magnitudes.

#### 3.3.6 Failure points

- **Humanised text.** As above: the dominant failure mode, and the one vendors spend most of their retraining budget on.
- **Long-form structure.** Sentence-level ensemble scoring does not model document architecture, so outline-generated symmetry (§2.3.2) is largely invisible to it.
- **Domain shift.** A model trained on web prose and marketing copy is not calibrated for organic chemistry lab reports.
- **Marketing/independent gap.** The vendor's headline numbers and the independent numbers differ by double digits, which is normal and should be assumed for every vendor in this table.

---

### 3.4 Pangram Labs

#### 3.4.1 Position

Pangram is the newest of the four and the most explicitly research-driven. Founded in 2023, it entered a market already crowded with detectors and differentiated on a single axis: **performance in the extreme low-false-positive regime**, with published confidence intervals and a stated scope of validity. The company's technical blog posts read like lab notes — they publish the corpus sizes, the confidence intervals, and the failure cases, including the ones that make their own product look worse.

#### 3.4.2 Architecture

Pangram describes a transformer encoder with a softmax classification head, trained on the order of **one million documents**, and applied over **overlapping windows of roughly 512 tokens** with a stride that ensures no token is scored without left and right context:

```text
for each window w_j of 512 tokens with stride s:
    h_j      = Encoder(w_j)
    p_j(t)   = softmax(W h_j(t) + b)      per-token distribution
score_doc    = Aggregate( { p_j(t) } )    with smoothing across overlapping windows
```

The per-token output is **three-way**: human, AI-assisted, AI-generated. The distinction between "AI-assisted" and "AI-generated" is the same insight that drove GPTZero's `L1` split and Turnitin's 2024 category change, arrived at independently: mixed authorship is not a rounding error, it is the modal case in professional and academic writing, and a binary classifier mislabels it in both directions.

Pangram 4 replaced the single encoder with a **mixture-of-experts** architecture at roughly six times the parameter count. The stated motivation is capacity for domain diversity rather than raw accuracy: a single dense model trained on essays, news, and technical documentation learns an average decision boundary, while an MoE can route different registers to different experts. Whether the routing actually captures register is not publicly documented, but the design is consistent with the observed improvement on out-of-domain benchmarks.

#### 3.4.3 Training methodology: mirrored prompts and hard negatives

Two methodological choices explain most of Pangram's published numbers.

**Mirrored prompts.** For a large fraction of the training corpus, human and machine documents are generated from the *same* prompt. This controls for topic, genre, and length — the confounders that make naive human-vs-AI corpora so easy to classify. A model trained on unmatched corpora can learn "this document is about chemistry" as a proxy for "this document is human," because the human documents happened to be chemistry papers. Mirrored prompts force the classifier to attend to style rather than subject matter.

**Hard-negative mining.** The training set is iteratively augmented with the examples the current model gets wrong: humanised machine text, lightly edited human text, translated text, and — importantly — human text written by non-native speakers and by neurodivergent writers. This is the standard remedy for a classifier that has learned a spurious correlate, and it is expensive, because each mining round requires generating and labelling new adversarial data.

```text
D_0 = base corpus (mirrored prompts)
repeat k times:
    hard_negatives = { x : x ∈ candidate pool, model(x) confidently wrong }
    D_{k+1} = D_k ∪ hard_negatives
    retrain with class-balanced loss
```

#### 3.4.4 Published performance

Pangram's numbers are the most precisely specified in the category, which is itself a signal about how the company wants to be evaluated:

| Metric | Value | Corpus / condition |
| --- | --- | --- |
| False-positive rate | **0.0041%** (CI 0.0032–0.0050) | Large human corpus, document level |
| False-negative rate | **0.3396%** | 519,993 documents, 26 generators |
| Humaniser detection | **98.83%** | 13 commercial humanising tools |
| Poetry FPR | 0.078% | Poetry corpus (a known weak spot) |
| Diverse-essay FPR | **0 false positives** | 25,996 essays |
| Neurodivergent-writer FPR | 0.0011% | Dedicated study |
| Short-text performance | Degrades below ~50 words | Length-stratified table |

The 0.0041% FPR is the headline and it is genuinely extraordinary: at that rate, screening 100,000 human documents yields about four false positives. But the number is only interpretable alongside its **denominator and its operating point**. It is a document-level FPR at a threshold chosen to maximize recall subject to that constraint, on a corpus the vendor assembled. It does not mean that any given flagged document is 99.996% likely to be AI — that is the base-rate error of §1.6.2, and Pangram's own blog addresses it directly, which is rare and creditable.

The FNR of 0.34% across 26 generators is the more surprising figure, because cross-generator generalization is where every other system degrades. If accurate, it implies the model learned something closer to a general "machine-ness" boundary than a set of model-specific fingerprints. The most plausible mechanism is the hard-negative mining regime, which repeatedly punishes reliance on any single generator's idiosyncrasies.

#### 3.4.5 Scope and stated limits

Pangram publishes an explicit out-of-scope list, which is unusual and analytically valuable:

- Very short texts (below roughly 50 words) — insufficient signal.
- Code, mathematical notation, and structured data — not natural language in the relevant sense.
- Heavily edited or collaboratively written text — mixed authorship is intrinsically ambiguous.
- Non-English text — the model is English-centric, and cross-lingual performance is not claimed at the same level.
- Deliberately adversarial paraphrase by a skilled human editor — no statistical detector survives this reliably.

The scope list is more informative than the performance table, because it tells you where the vendor believes the measurement is valid. A detector that states its domain of validity can be used responsibly; one that claims universality cannot.

#### 3.4.6 Failure points

- **Cost and latency.** Windowed inference over a 6× larger MoE is expensive relative to a single-pass encoder, which constrains throughput for high-volume platforms.
- **Domain novelty.** A new genre (say, legal drafting or clinical notes) is out of distribution until it appears in the training mix.
- **The adversarial ceiling.** The 98.83% humaniser detection is measured against 13 known tools. A determined human paraphraser is not in that set, and no published system claims robustness against one.
- **Interpretation risk.** Extremely low FPRs encourage over-trust. A 0.0041% FPR still produces false positives, and those false positives are concentrated in exactly the populations — non-native writers, neurodivergent writers, formulaic genres — that the hard-negative mining was meant to protect.

---

### 3.5 Cross-Vendor Comparison

| Dimension | Turnitin | GPTZero | Originality.ai | Pangram |
| --- | --- | --- | --- | --- |
| Unit of analysis | Sentence (contextual) | Document + sentence | Sentence + document | Token → window → document |
| Aggregation | % of qualifying sentences above threshold | Learned document head + smoothing | Ensemble combiner | Smoothed window aggregation |
| Label space | AI / not-AI (paraphrase split 2024) | Human / AI / Mixed; Pure / Polished / Paraphrased | AI / human / paraphrased | Human / AI-assisted / AI-generated |
| Minimum length | 300 words | Short-text degraded, no hard floor | ~100 words | ~50 words (degraded below) |
| Languages | 7 | Multilingual claim | Primarily English | Primarily English |
| Claimed document FPR | <1% | Not stated as a single figure | <1% at operating point | 0.0041% |
| Paraphrase defence | Dedicated since 2023 | 93.5% bypasser recall (vendor test) | Dedicated classifier | 98.83% over 13 tools |
| Process evidence | None | Writing Replay | None | None |
| Independent benchmark presence | Weber-Wulff; institutional studies | Limited | RAID; *Eye* study | Limited but public methodology |
| Primary weakness | Formulaic academic prose; short work | Polished human text | Humanised text | Cost; domain novelty |

#### 3.5.1 What all four share

Beneath the marketing differences, the systems are architecturally convergent:

1. **A supervised transformer encoder is the core.** None of the four relies primarily on perplexity, burstiness, or any closed-form statistic. All four use learned representations, and at least three explicitly deny that perplexity is a primary signal (Turnitin's FAQ states the model "is not explicitly programmed to evaluate specific signals such as 'burstiness,' 'perplexity'").
2. **Windowed or sentence-level scoring plus an aggregation policy.** The unit of inference is smaller than the document because localisation is what users want and because long documents exceed context windows.
3. **A post-processing layer that suppresses low-confidence output.** Asterisks, confidence tiers, smoothing, and thresholds all exist to convert a continuous score into a defensible claim.
4. **Adversarial retraining on paraphrased and humanised text**, with a documented accuracy cost on in-distribution data.
5. **A mixed-authorship category added after launch**, because the binary framing failed in practice.
6. **No disclosure of the training corpus**, which makes independent replication impossible and makes every published accuracy figure a claim about an undisclosed distribution.
7. **A retraining cadence tied to model releases**, which means the product's performance is a moving target and any benchmark older than about six months is historical.

#### 3.5.2 Where they genuinely diverge

- **Operating point.** Turnitin accepts a higher FPR to catch more AI text at institutional scale, because its output feeds a human review process. Pangram accepts lower recall to achieve a four-orders-of-magnitude lower FPR, because its customers use the score as a decision rather than a prompt. Both are rational; they are not comparable.
- **Explanatory surface.** GPTZero invests in saliency and sentence highlighting because its users are teachers who need to justify a conversation. Turnitin invests in report integration because its users are administrators who need a workflow.
- **Data strategy.** Originality's moat is a web-scale index; Pangram's is a curated adversarial corpus; Turnitin's is a decade of institutional submissions; GPTZero's is a large consumer user base generating diverse prompts.
- **What counts as evidence.** Only GPTZero ships process evidence (Writing Replay). The others are text-only, which caps what any of them can prove.

#### 3.5.3 How to read a vendor accuracy claim

A seven-question checklist that separates a meaningful number from a marketing one:

| Question | Why it matters |
| --- | --- |
| What is the unit — sentence or document? | Sentence-level FPRs look far better than document-level ones |
| At what operating point? | A TPR at 10% FPR is not a TPR at 0.01% FPR |
| On what corpus, and is it disjoint from training? | Vendor corpora are usually in-distribution |
| Were unseen generators included? | Cross-model generalization is the real test |
| Was an adversary in the loop? | Unmodified AI text is the easy case |
| Is there a confidence interval? | A 0.0041% FPR on 10,000 documents is not estimable |
| What is the base rate in deployment? | Precision, not accuracy, determines usefulness (§1.6.2) |

#### 3.5.4 The independent-evaluation gap

The most important structural fact about this market is that **almost no vendor has been independently audited at its claimed operating point.** The available independent evidence is thin and sobering:

- [Weber-Wulff et al. (2023)](https://arxiv.org/abs/2306.15666) tested 14 detection tools on a controlled corpus and found that **none exceeded 80% accuracy**, with only five above 70%; the authors concluded that the tools were "not suitable" for decisions about academic misconduct.
- [RAID (Dugan et al., ACL 2024)](https://arxiv.org/abs/2405.07940) evaluated 8 open and 4 closed detectors across 6 million generations from 11 models, 8 domains, 11 adversarial attacks, and 4 decoding strategies, and found that all were "easily fooled" by adversarial attacks, sampling-strategy variation, repetition penalties, and unseen generators.
- [Sadasivan et al. (2023)](https://arxiv.org/abs/2303.11156) give the theoretical version of the same result: for any detector, there exists a paraphraser that drives the total-variation distance between human and machine distributions below any threshold, so robust detection is not achievable without either a watermark or a limitation on the adversary's compute.

These results do not contradict the vendors' numbers. They measure different things: vendor benchmarks measure in-distribution discrimination, and independent benchmarks measure out-of-distribution robustness. Both can be true. The professional inference is that **a detector's score is evidence about statistical typicality, and evidence about typicality is not evidence about authorship** — a distinction the next section develops into a decision framework.

---

## 4. Limitations, Obfuscation, and the False Positive Dilemma

The first three sections described what detectors measure, what the text looks like, and how the commercial systems are built. This section describes why the measurement cannot be made reliable in the general case, why its errors are not randomly distributed, and what a defensible use of the technology would look like given those constraints.

The argument has four parts. **Statistical**: detection is a two-sample test under distribution shift, and the shift is controlled by an adversary. **Distributional**: the false-positive burden falls on writers whose prose is formulaic, non-native, or genre-conforming. **Adversarial**: paraphrasing attacks the channel detectors rely on most, and the defence is expensive and partial. **Institutional**: the decision-theoretic threshold for a disciplinary action is far above any detector's achieved precision.

---

### 4.1 The Statistical Limit: Why Robust Detection Is Not Achievable

#### 4.1.1 The separation bound

Section 1.4 introduced Le Cam's bound. Here is its operational form. For any detector `A` with threshold `τ`, the difference between its true- and false-positive rates is bounded by the total variation distance between the human and machine distributions:

```text
TPR(τ) − FPR(τ)  ≤  d_TV( P_human , P_AI )
```

This is not a statement about engineering quality. It is a property of hypothesis testing: if two distributions overlap substantially, no decision rule can separate them, and the best achievable accuracy is `½ + ½·d_TV`. A detector that reports 99% accuracy on a benchmark is reporting that, *on that benchmark*, the two distributions are far apart. The benchmark's separation is a property of the benchmark.

The consequence is that every detector's performance is a statement about a specific pair of distributions, and both members of the pair move. `P_AI` moves with every model release. `P_human` moves as writers adapt — and, importantly, as human writers absorb the LLM lexicon through exposure, which is now measurable at population scale (§2.2.2).

#### 4.1.2 Recursive paraphrasing and the collapse toward chance

[Sadasivan et al. (2023)](https://arxiv.org/abs/2303.11156) make the adversarial version of the argument precise. They define a recursive paraphrasing attack: pass the machine text through a paraphraser, then through the detector, then paraphrase again, iterating. Each pass reduces the total variation distance between the machine text and the human distribution, because paraphrasing is trained to produce fluent, human-like text. The bound above then forces the detector's achievable separation down:

```text
d_TV( P_human , T(P_AI) )  <  d_TV( P_human , P_AI )        for paraphrase operator T
⇒  TPR − FPR  ↓  as T is applied recursively
```

Their empirical result is that detection rates fall sharply under recursive paraphrasing while text quality degrades only slightly — the key asymmetry. An attacker can afford to lose a little quality; a detector cannot afford to lose separation. They also demonstrate a **spoofing attack**: an adversary can infer enough of a watermark's signature to make human text trigger it, which converts a false-negative problem into a false-positive problem and turns the watermark against its users.

The strongest empirical confirmation comes from the [DIPPER](https://arxiv.org/abs/2303.13408) paraphrase model: paraphrasing dropped DetectGPT's detection rate from **70.3% to 4.6% at a constant 1% false-positive rate**, "without appreciably modifying the input semantics." That last clause is the crux. The attack preserves meaning — so every semantic and structural signal survives — while destroying the token-level statistics the detector was reading. The authors' defence is equally instructive: a **retrieval-based** check that compares a candidate document against a database of previously generated outputs, which recovered **80–97%** of paraphrased generations at a 1% FPR. But the defence requires the model provider to retain every generation and expose a lookup API. It is an architectural solution, not a statistical one, and it only works inside a closed ecosystem.

#### 4.1.3 Watermarking: the only structural escape

A watermark solves the problem by changing the generative process rather than the detector. If the model samples from a keyed distribution, provenance becomes a property of the token stream with a formal false-positive guarantee ([Kirchenbauer et al., 2023](https://arxiv.org/abs/2301.10226)):

```text
s_t ~ softmax( l_t + δ · 1[ token ∈ G_t ] ),   G_t = hash( key, s_{t−1} )
z   = ( |s|_G − γ T ) / sqrt( T γ (1 − γ) )
p   = 1 − Φ(z)
```

At `z = 4`, the p-value is about `3 × 10⁻⁵` — a guarantee no statistical detector can offer. The scheme is also fragile in exactly the ways that matter commercially: it requires the generator to cooperate; it is unavailable for open-weight models, which anyone can fine-tune or re-sample; it does not survive paraphrase, translation, or substantial editing; and it is detectable and removable by an adversary with white-box access.

The policy environment has moved toward mandating provenance rather than detection. The EU AI Act's transparency provisions (Article 50, applicable from August 2026) require machine-readable marking of synthetic content, and C2PA Content Credentials provide a container format for cryptographic provenance. Neither addresses the case that dominates academic integrity — a student typing a model's output into a word processor — because the marking lives in metadata the editor discards.

The honest summary is that **there are two viable regimes and no third**: closed ecosystems with watermarking and generation logs, where provenance is near-certain, and open ecosystems with statistical detection, where provenance is probabilistic and adversarially degradable. Every commercial detector in Section 3 operates in the second regime while being marketed as if it were in the first.

---

### 4.2 The False Positive Dilemma: Base Rates, L2 Bias, and Disparate Impact

#### 4.2.1 The mechanism of second-language bias

The bias is not a bug in a particular model. It is a predictable consequence of what the detectors measure.

Detectors read low surprisal as evidence of machine generation. Second-language writers produce systematically lower surprisal in English, for reasons that have nothing to do with machines:

- **High-frequency vocabulary.** L2 writers use a narrower, more frequent word set, because rare words are riskier. Rare-word density is a standard feature in detector ensembles.
- **Formulaic phrasing.** Instructional settings teach explicit templates ("In my opinion…", "Firstly… Secondly…", "On the other hand…"). Templates are, by construction, high-probability sequences.
- **Repetition.** Lexical repetition is a normal compensation strategy when synonyms are unavailable. Detectors read it as low burstiness.
- **Syntactic conservatism.** Shorter sentences, simpler subordination, fewer sentence-final flourishes — the same profile as default model output.
- **Register mismatch.** Many L2 writers are trained on textbook English, which is closer to the "edited English" basin that preference-optimised models occupy than to idiomatic native prose.

Each of these is a rational adaptation by the writer. Together they push the document toward the machine side of the decision boundary. The detector is not measuring authorship; it is measuring **proficiency in idiomatic English**, and then interpreting low proficiency as machine origin.

#### 4.2.2 The evidence

[Liang et al. (2023)](https://arxiv.org/abs/2304.02819) evaluated several widely used GPT detectors on writing samples from native and non-native English writers. Their finding is unambiguous: the detectors "consistently misclassify non-native English writing samples as AI-generated," while native samples are accurately identified. Across the detectors tested, an average of **61.3%** of the non-native essays were flagged — and for some detectors the rate exceeded 90%. The control set of native-speaker essays was flagged at a small fraction of that rate.

Two further results from that study matter:

1. **Simple prompting mitigates the bias.** Asking a model to "revise the text to be more formal" or to use simpler language moved non-native samples across the boundary in both directions. This shows the signal is shallow — it responds to register manipulation rather than to authorship.
2. **The same prompting bypasses detectors.** The intervention that reduces false positives is also an evasion technique. This is a structural feature of the problem: any transformation that makes human text look less machine-like also makes machine text look less machine-like.

The pattern replicates in adjacent populations. Neurodivergent writers, writers using AAC devices, and writers producing formulaic professional prose show elevated flag rates for the same distributional reasons. Pangram's decision to run a dedicated neurodivergent-writer study and report a 0.0011% FPR is a tacit acknowledgement that this population is a known failure mode — and the fact that a vendor had to run that study is itself evidence about the baseline.

#### 4.2.3 The arithmetic of accusation

Consider an institution screening 30,000 submissions with a detector operating at a 1% document-level FPR — better than most vendors claim.

```text
Submissions                     30,000
Human-authored (estimated 85%)  25,500
False positives @ 1%               255
True AI documents (15%)          4,500
True positives @ 85% TPR          3,825
Total flags                      4,080
Precision (PPV)                   93.8%
```

A 93.8% precision sounds acceptable until the errors are disaggregated. If the false-positive rate is not uniform — say 0.3% for native writers and 5% for L2 writers, consistent with the literature — then:

```text
L2 submissions (15% of total)     4,500
L2 false positives @ 5%             225
Native false positives @ 0.3%        77
L2 share of all false positives     74.5%
Relative risk of false flag         16.7×
```

Three-quarters of the wrongly accused students come from 15% of the population. That is the definition of disparate impact, and it is produced by an error rate the vendor is entitled to describe as "under 1%."

The downstream institutional cost compounds. Each false positive consumes an investigation, damages a relationship, and — if the accusation is made public or recorded — imposes a cost that is not commensurable with the cost of a missed detection. This asymmetry is the subject of §4.5.

#### 4.2.4 Why vendor counters do not settle it

Turnitin reports a false-positive rate of **0.014 for ELL writers versus 0.013 for non-ELL writers**, which appears to contradict the independent literature. Several things can be true at once:

- The vendor measures its own corpus, with its own definition of "ELL," at its own operating point, on documents above 300 words.
- Its model has been explicitly tuned to reduce this specific error since 2023, precisely because the literature identified it.
- The independent studies use short TOEFL-style essays, which are both shorter and more formulaic than the documents Turnitin accepts.
- Vendor evaluation samples are rarely drawn from the population that generates complaints.

The correct reading is not that either side is lying, but that **a vendor's FPR estimate is an in-distribution estimate, and the populations at risk are out of distribution by construction.** The number that matters to a dean is not the vendor's average; it is the rate among the students most likely to be wrongly flagged, and that number is not published by anyone.

#### 4.2.5 Where the dilemma actually bites

| Population | Mechanism | Consequence |
| --- | --- | --- |
| L2 / international students | Low-surprisal formulaic prose | 10–60× relative risk of false flag |
| Neurodivergent writers | Repetitive structure, literal phrasing | Elevated FPR; rarely measured |
| Students in formulaic genres | Lab reports, reflections | Genre-level false positives |
| Students who use Grammarly | Surface normalisation | Raises "polished" score |
| Students who legitimately use AI for brainstorming | Mixed authorship | Indistinguishable from misconduct |
| Students with old hardware / no keystroke logs | No process evidence | No exculpatory record |

The last row is the one institutions least often consider. The students most likely to be accused are also the least likely to possess the version history, cloud backups, or drafting artefacts that would exonerate them. The evidentiary asymmetry runs in the same direction as the statistical one.

---

### 4.3 The Low-Perplexity Trap: Technical and Academic Writing

#### 4.3.1 Formulaic prose is machine-like by construction

Academic and technical writing is the genre most exposed to false positives, because its conventions were optimised for density and reproducibility rather than for stylistic variance.

- **Terminology is fixed.** A paper about transformer attention must use "attention," "query," "key," "value," "softmax" repeatedly. Repetition is correct, not suspicious.
- **Nominalisation is the register.** "The implementation of the protocol resulted in a reduction in latency" is *good* technical prose and *bad* perplexity. The nominal style that instruction-tuned models overproduce (§2.2.1) is the style that technical writing manuals teach.
- **Section structure is standardised.** Abstract, introduction, related work, method, results, discussion, conclusion — in that order, at roughly those proportions. Outline-generated symmetry (§2.3.2) is indistinguishable from disciplinary convention.
- **Hedging is mandatory.** "These results suggest," "may indicate," "under certain conditions" — the epistemic caution that detectors read as globally symmetrical hedging is a peer-review requirement.
- **Boilerplate is inherited.** Ethics statements, data-availability statements, funding acknowledgements, and reproducibility checklists are near-verbatim across thousands of papers, human and machine alike.

#### 4.3.2 The conformity meter

There is a sharper way to state the problem. A detector trained to separate human from machine academic prose is trained on a corpus in which the human class is *also* constrained by genre. What the classifier can learn, therefore, is not "human versus machine" but "**conforms to the genre norm versus deviates from it**" — because the machine class was trained to produce the norm, and the human class contains both conforming and deviating examples.

```text
score(x)  ≈  α · conformity(x, genre)  +  β · machine_fingerprint(x)  +  ε
```

For an expert writer who follows convention precisely, `conformity` is high and the score rises. For a distinctive stylist, it falls. The detector is, in part, a **conformity meter**, and conformity is correlated with training, discipline, and cultural background rather than with authorship.

This predicts a counter-intuitive result that the empirical literature confirms: **better writers are more likely to be flagged in technical genres**, because the features that make technical prose good are the features the model was optimised to produce. The "expertise penalty" is not a paradox; it is what happens when the target class is defined by adherence to a norm that the model has learned to reproduce.

#### 4.3.3 Genre exposure

| Genre | Typical human perplexity | Typical human burstiness | False-positive risk | Why |
| --- | --- | --- | --- | --- |
| Personal essay | Moderate | High | Low | Idiolect, narrative, digression |
| Literary fiction | High | High | Low–moderate | Deliberate deviation from norm |
| News report | Low–moderate | Moderate | Moderate | Formulaic inverted pyramid |
| Lab report / methods | Low | Low | **High** | Nominalisation, passive, repetition |
| Systematic review | Low | Low | **High** | Structured abstract, boilerplate |
| Grant application | Low | Low | High | Persuasive formula, evaluative adjectives |
| Legal contract | Very low | Very low | **Very high** | Recursive boilerplate, defined terms |
| Poetry | High | High | Moderate | Short, allusive; Pangram reports 0.078% FPR |
| Code documentation | Low | Low | High | Imperative templates, parameter lists |

The table explains why vendor benchmarks built on student essays systematically overstate field performance. Student essays are the easy genre: high variance, personal voice, narrative structure. The documents that actually generate disputes are lab reports and review articles, which sit closest to the decision boundary.

#### 4.3.4 The feedback loop

There is a slow-moving failure mode that no vendor can fix. Population-level estimates of LLM use in academic writing — Kobak's 13.5%, Liang's 17.5% — are computed from **vocabulary shift**. They work because human academic vocabulary was stable before 2023 and jumped afterward. But as human writers absorb the new lexicon (through reading, through editing tools, through imitation of published prose), the human distribution shifts toward the machine distribution. The signal that made the estimate possible decays with every paper published in the new register.

The same dynamic affects detectors. Each retraining cycle fits a boundary between two distributions that are converging. The convergence is not hypothetical: instruction-tuned models are trained on human feedback, and human feedback is now substantially informed by text that models produced.

---

### 4.4 Obfuscation: Paraphrase, Humanisers, and Style Laundering

#### 4.4.1 Taxonomy of transformations

| Transformation | Mechanism | Channel attacked | Residual signal | Cost / accessibility |
| --- | --- | --- | --- | --- |
| Synonym substitution (thesaurus, QuillBot "Standard") | Word-level replacement under a fluency constraint | T1 lexical, partially T2 | Sentence length, structure, entailment, grounding | Seconds; free |
| Sentence restructuring / voice change | Clause reordering, active↔passive | T1, T2 rhythm | Discourse order, section symmetry | Seconds; free |
| Back-translation (EN→X→EN) | Pivot through another language | T1 idiom, tokeniser artefacts | Content, architecture; adds translationese | Minutes; free |
| Paraphrase LM (DIPPER-class) | Conditional regeneration with lexical/reorder control | T1 strongly; watermark destroyed | T2/T3 intact — semantics preserved by design | Minutes; open weights |
| Commercial humaniser | Adversarially trained rewriter | T1, some T2 | Vendor-dependent; now a detection target | Subscription |
| Invisible characters / homoglyphs | Zero-width joiners, Cyrillic lookalikes | Tokenisation, matching | Trivially normalised away | Seconds |
| Adversarial suffix (GCG-class) | Gradient-optimised token append | Model-specific classifiers | Suffix is incoherent; detectable by perplexity | High compute |
| Manual line edit | Human rewriting pass | All surface channels | T3 unless content is added | Hours |
| Content injection | Add real, specific, checkable detail | T3 | None — this is authorship | Hours; requires knowledge |
| Hybrid drafting | Human outline → model expansion → human edit | Mixed | Mixed, per-sentence | The realistic case |

#### 4.4.2 Why paraphrase works, and where it stops

Paraphrase works because it is a **surface-channel attack on a system whose most heavily weighted features are surface-level.** Modern detectors are transformer classifiers trained on token sequences; the gradient of their decision function is largest along lexical and syntactic directions, because those are the directions in which the training corpora differ most. A paraphraser moves the document along exactly those directions.

It stops working at the point where the detector models **information flow** rather than surface form. Sentence-length variance can be restored by a paraphraser that varies clause length. Lexical tells can be removed by substitution. But a paraphraser trained to preserve meaning cannot:

- change how many sections the document has,
- make the sections unequal in length,
- reduce the entailment closure between consecutive paragraphs,
- introduce a verifiable referent that was not in the source,
- or give the document a stance that varies with the argument's confidence.

This is why the vendors that report the strongest anti-humaniser numbers are the ones that describe their models in terms of document-level and token-level context rather than word lists. GPTZero's 93.5% bypasser recall and Pangram's 98.83% across 13 tools are plausible precisely because they are measured against *known* tools whose transformations are bounded; a human editor with an hour is a different adversary, and no vendor claims robustness against that.

#### 4.4.3 The cure has the disease

There is an irony that deserves explicit statement. Paraphrase models are themselves language models, trained on the same corpus, optimised for the same fluency objective. Their output therefore carries **its own machine signature**: nominalisation, uniform clause structure, and the same preference-optimised lexicon. A document that has been "humanised" is not human text; it is *twice-machine text* with a perturbed surface.

Three consequences follow:

1. **Detectors trained on humanised data recover most of the lost accuracy.** This is exactly what adversarial training does, and it is why the vendor numbers have climbed back into the 90s after the 2023 collapse.
2. **Humanised text is stylistically homogeneous in a new way.** It tends to be *more* uniform than the original, because the paraphraser normalises toward its own mode. A detector that measures variance may find a humanised document *more* suspicious on T2 features even as it looks less suspicious on T1.
3. **The attack has a semantic ceiling — but not a strategic one.** A paraphraser must preserve meaning to be useful; a detector only needs a statistical trace, and traces accumulate across a document. That favours the detector on any *single* transformation. It does not follow that the defender wins, because the attacker can iterate and iteration is cheap (§4.4.4). The humaniser market is a subscription treadmill for both sides.

#### 4.4.4 The economics of the arms race

Model the adversary as someone who can attempt `k` independent transformations and stops at the first success. If each attempt defeats the detector with probability `p`, the probability of at least one success is:

```text
P(evade) = 1 − (1 − p)^k
```

With `p = 0.3` per attempt, three attempts yield a 65.7% evasion probability; ten attempts yield 97.2%. The attacker's cost is linear in `k` and mostly automated; the defender's cost is a full retraining cycle per new attack family, plus the accuracy cost of adversarial training on in-distribution data (documented by GPTZero and by the adversarial-robustness literature generally). This is a classic asymmetric security problem, and it has the classic asymmetric outcome: **the defender must be right about every attack; the attacker needs one.**

The DIPPER result quantifies the asymmetry: a single paraphrase pass took DetectGPT from 70.3% to 4.6%. The retrieval defence restored 80–97%, but only for a provider that logs every generation — which is to say, only for the largest closed platforms, and only for text generated through their API rather than copied from a chat window.

#### 4.4.5 "Style laundering"

The realistic obfuscation pipeline is not a single tool. It is a chain:

```text
human prompt → LLM draft → paraphrase → back-translation → human edit → submission
```

Each stage removes a different surface trace. The LLM draft supplies structure and content. The paraphraser removes the lexical fingerprint. Back-translation removes idiomatic tells and introduces mild translationese, which further disrupts perplexity. The human edit restores local voice and fixes the artefacts that the earlier stages introduced.

What survives is the **information architecture**: the section symmetry, the entailment closure, the absence of verifiable referents, the uniform stance. These are the properties that no stage in the chain was designed to change, because changing them would require rewriting the argument. The chain is, in the precise sense, a **style-laundering operation**: it cleans the surface while leaving the structure intact.

This is the strongest argument for the T2/T3 approach and the strongest argument against the current generation of commercial detectors, most of which still weight surface features heavily because surface features are cheap to learn. A detector that models document architecture is harder to build and harder to fool, and it is also harder to explain to a user — which is why the market has been slow to adopt it.

---

### 4.5 A Calibrated Evidence Framework

#### 4.5.1 An evidence hierarchy

The binary question "is this AI?" is the wrong question, because it has no reliable answer. The useful question is "what does the evidence support, and what action does that justify?" The following hierarchy orders evidence by probative value:

| Level | Evidence | What it establishes | Evidential strength |
| --- | --- | --- | --- |
| 0 | Provenance metadata, watermark, generation log | Generation by a specific system | Conclusive, when intact |
| 1 | Process evidence (version history, keystroke replay) | Composition process | Strong, direct |
| 2 | Referential verification (nonexistent citations, impossible dates, fabricated data) | Fabrication | Strong for misconduct, not for authorship |
| 3 | Detector score + known operating point + base rate | Statistical typicality | Moderate, probabilistic |
| 4 | Stylometric anomaly (Delta, variance ratios) | Deviation from an author's baseline | Weak alone |
| 5 | Lexical tells ("delve," em dashes) | Nothing, in isolation | Negligible |

The hierarchy has a practical implication that is routinely ignored: **levels 0–2 dominate levels 3–5, and levels 3–5 should never be used without them.** A detector score is not a weaker version of a watermark; it is a different kind of claim. It says "this text resembles the machine class" and not "this text was produced by a machine."

#### 4.5.2 The decision-theoretic threshold

Suppose the posterior probability that a document is machine-generated, given all evidence, is `p`, and the institution must decide whether to act. Let `C_FP` be the cost of a wrongful accusation and `C_FN` the cost of a missed case. Acting is justified when:

```text
p · C_FN  >  (1 − p) · C_FP
⇒  p*  =  C_FP / (C_FP + C_FN)
```

| Context | `C_FP : C_FN` | Required posterior `p*` | Practical meaning |
| --- | --- | --- | --- |
| Formative feedback ("revise this") | 1 : 1 | 0.50 | Any weak signal is enough to prompt a conversation |
| Request for a draft history | 5 : 1 | 0.83 | Needs a clear detector signal plus context |
| Formal misconduct referral | 50 : 1 | **0.98** | Needs corroborating evidence beyond a score |
| Public finding / sanction | 500 : 1 | **0.998** | Needs conclusive provenance or admission |

No commercial detector's document-level posterior reaches 0.98 on a contested document at realistic base rates. That is not a criticism of the vendors; it is arithmetic. The vendors' own guidance — Turnitin's suppression below 20%, Originality's warning that false positives are "still too high to be relied upon for disciplinary action," Pangram's published scope limits — is consistent with this conclusion.

The framework also explains why the same detector can be legitimately used in one context and illegitimately in another. Where `p* = 0.5` — a formative prompt to revise — a weak signal is enough to justify a conversation, and a 20% score is actionable. Where `p* = 0.98` — a formal misconduct finding — the same score is not evidence at all, because the posterior it supports is nowhere near the threshold.

#### 4.5.3 An institutional protocol

A defensible protocol follows directly from the hierarchy and the threshold:

1. **Never adjudicate on a score alone.** Require at least one piece of level 0–2 evidence before any formal process.
2. **Preserve process artefacts.** Encourage version history, drafts, annotated bibliographies, and in-class writing. Where these exist, they dominate the statistical evidence.
3. **Screen, do not accuse.** Treat a flag as a reason to look, not a finding. The first communication should be a request for the drafting record, not an allegation.
4. **Calibrate to the population, not the benchmark.** If the institution's population includes many L2 writers, the effective FPR is higher than the vendor's headline, and the threshold must rise accordingly.
5. **Publish the error rates.** Students are entitled to know that the tool has a non-zero false-positive rate, that the rate is unevenly distributed, and that a flag is not proof.
6. **Prefer assessment design to detection.** Oral defence, in-class drafting, scaffolded assignments with visible drafts, and discipline-specific writing instruction all produce better evidence than any classifier — and they improve learning rather than policing it.
7. **Revisit annually.** Every model release changes the operating characteristics. A policy calibrated in 2024 is not calibrated in 2026.

#### 4.5.4 Concluding assessment

Four conclusions survive the analysis.

**Detection works best where it matters least.** Accuracy is highest on long, unedited, single-author, English, non-technical documents produced by a model in the detector's training set. Those are precisely the documents least likely to be disputed. The disputed cases are short, edited, mixed, non-native, technical, or produced by a model released last month — the corner of the space where every published evaluation degrades.

**The false-positive burden is structural, not incidental.** It falls on writers whose prose is formulaic: second-language writers, neurodivergent writers, students in lab-report genres, and experts conforming to disciplinary convention. No amount of retraining removes this, because the mechanism is the same mechanism that makes detection possible: low surprisal is read as machine origin, and low surprisal is also what instruction and convention produce.

**Obfuscation defeats the shallow channels and cannot defeat the deep ones.** Paraphrase, back-translation, and humanisers reliably destroy token-level statistics, which is why vendor accuracy collapsed in 2023 and recovered only through adversarial retraining. They cannot change section symmetry, entailment closure, or referential grounding without changing the argument. The durable signal is architectural, and the detectors that weight architecture are the ones whose numbers have held up.

**The technology is a screening instrument, not an oracle.** Used as a screening instrument — at a low threshold, with human review, with process evidence, and with the base rate made explicit — it can help an institution notice something it might have missed. Used as an oracle, it produces exactly the disparate outcomes its own error rates predict, and it does so with the appearance of objectivity. The difference between those two uses is not a matter of engineering. It is a matter of institutional discipline, and no vendor can supply it.

---

### 4.6 Reference Register

| # | Work | Venue / year | Link | Used for |
| --- | --- | --- | --- | --- |
| 1 | "Artificial intelligence content detection" | Wikipedia (accessed Oct 2026) | [link](https://en.wikipedia.org/wiki/Artificial_intelligence_content_detection) | Survey of methods, tool performance, ESL findings |
| 2 | "Perplexity" | Wikipedia (accessed Oct 2026) | [link](https://en.wikipedia.org/wiki/Perplexity) | Definition, Brown Corpus benchmark |
| 3 | Brown, Della Pietra, Mercer, Della Pietra & Lai, "An Estimate of an Upper Bound for the Entropy of English" | *Computational Linguistics* 18(1), 1992 | [link](https://aclanthology.org/J92-1003/) | 1.75 bits/char; 247 per word; trigram model |
| 4 | Gehrmann, Strobelt & Rush, "GLTR: Statistical Detection and Visualization of Generated Text" | ACL 2019 | [link](https://arxiv.org/abs/1906.04043) | Rank-based token tests |
| 5 | Mitchell, Lee, Khazatsky, Manning & Finn, "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature" | ICML 2023 | [link](https://arxiv.org/abs/2301.11305) | Log-probability curvature |
| 6 | Bao, Zhao, Teng, Yang & Zhang, "Fast-DetectGPT" | ICLR 2024 | [link](https://arxiv.org/abs/2310.05130) | Conditional probability curvature |
| 7 | Hans, Schwarzschild, Cherepanova, Kazemi, Saha, Goldblum, Geiping & Goldstein, "Spotting LLMs With Binoculars" | ICML 2024 | [link](https://arxiv.org/abs/2401.12070) | Observer/performer ratio; 0.01% FPR |
| 8 | Tulchinskii et al., "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts" | NeurIPS 2023 | [link](https://arxiv.org/abs/2306.04723) | Embedding geometry |
| 9 | Sadasivan, Kumar, Balasubramanian, Wang & Feizi, "Can AI-Generated Text be Reliably Detected?" | 2023 | [link](https://arxiv.org/abs/2303.11156) | TV bound, recursive paraphrase, spoofing |
| 10 | Kirchenbauer, Geiping, Wen, Katz, Miers & Goldstein, "A Watermark for Large Language Models" | ICML 2023 | [link](https://arxiv.org/abs/2301.10226) | Green-list watermark, z-test |
| 11 | Krishna, Song, Karpinska, Wieting & Iyyer, "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense" (DIPPER) | NeurIPS 2023 | [link](https://arxiv.org/abs/2303.13408) | 70.3%→4.6%; retrieval defence 80–97% |
| 12 | Liang, Yuksekgonul, Mao, Wu & Zou, "GPT detectors are biased against non-native English writers" | *Patterns*, 2023 | [link](https://arxiv.org/abs/2304.02819) | 61.3% mean FPR on TOEFL essays |
| 13 | Weber-Wulff, Anohina-Naumeca, Bjelobaba, Foltýnek, Guerrero-Dib, Popoola, Šigut & Waddington, "Testing of detection tools for AI-generated text" | *IJEI*, 2023 | [link](https://arxiv.org/abs/2306.15666) | 14 tools, none >80% accuracy |
| 14 | Kobak, González-Márquez, Horvát & Lause, "Delving into LLM-assisted writing in biomedical publications through excess vocabulary" | *Science Advances* 11(27), 2025 | [link](https://arxiv.org/abs/2406.07016) | 15M abstracts; ≥13.5% of 2024 abstracts |
| 15 | Liang et al., "Mapping the Increasing Use of LLMs in Scientific Papers" | 2024 | [link](https://arxiv.org/abs/2404.01268) | 950,965 papers; 17.5% CS / 6.3% math |
| 16 | Reinhart, Markey, Laudenbach, Pantusen, Yurko, Weinberg & Brown, "Do LLMs write like humans? Variation in grammatical and rhetorical styles" | *PNAS* 122, 2025 | [link](https://arxiv.org/abs/2410.16107) | Biber features; 2–5× participles; 1.5–2× nominalisations |
| 17 | Dugan et al., "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors" | ACL 2024 | [link](https://arxiv.org/abs/2405.07940) | 6M generations; 11 models; adversarial robustness |
| 18 | Burrows, "Delta: a Measure of Stylistic Difference and a Guide to Likely Authorship" | *LLC* 17(3), 2002 | [link](https://doi.org/10.1093/llc/17.3.267) | Stylometric distance |
| 19 | Biber, *Variation across Speech and Writing* | Cambridge UP, 1988 | [link](https://doi.org/10.1017/CBO9780511621024) | 66-feature tagset |
| 20 | Mosteller & Wallace, *Inference and Disputed Authorship: The Federalist* | Addison-Wesley, 1964 | [link](https://en.wikipedia.org/wiki/Federalist_Papers) | Function-word authorship attribution |
| 21 | "AI Writing Detection" FAQ and model documentation | Turnitin (accessed Oct 2026) | [link](https://www.turnitin.com/products/features/ai-writing-detection) | Sentence scoring, 20% floor, FPR, languages, formats |
| 22 | "GPTZero: Robust Detection of LLM-Generated Texts" | arXiv, 2026 | [link](https://arxiv.org/abs/2602.13042) | Ternary label space, joint loss, red teaming |
| 23 | GPTZero product and methodology documentation | GPTZero (accessed Oct 2026) | [link](https://gptzero.me) | Deep Scan, Writing Replay, API |
| 24 | Accuracy and model-release documentation | Originality.ai (accessed Oct 2026) | [link](https://originality.ai) | Ensemble architecture, TPR-at-FPR, version history |
| 25 | "How it works" and "Is Pangram accurate?" | Pangram Labs (accessed Oct 2026) | [link](https://www.pangram.com/research/how-it-works) | Architecture, hard negatives, scope limits |
| 26 | "Pangram 4" technical note | Pangram Labs, 2025 | [link](https://www.pangram.com/blog/pangram-4-technical) | MoE, 6× parameters, per-token classes |
| 27 | EditLens: "Quantifying the Extent of AI Editing in Text" | 2025 | [link](https://arxiv.org/abs/2510.03154) | Human-edited machine text; edit-distance stratification |
| 28 | "DAMAGE: Detecting Adversarially Modified AI Generated Text" | 2025 | [link](https://arxiv.org/abs/2501.03437) | Adversarially modified machine text |
| 29 | Regulation (EU) 2024/1689 (AI Act), Article 50 | *OJEU*, 2024 | [link](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) | Transparency and marking obligations |

*Vendor documentation is cited as primary evidence of what each system claims and how it describes its own pipeline; it is not treated as independent validation. Independent benchmarks are cited where they exist, and their limits — small samples, synthetic corpora, single-genre designs — are noted in the text.*

---

*End of document.*
