# Zero-shot / Statistical AI-Text Detectors and Text Watermarking: What They Measure and How Rewriting Affects Them

Research date: 2026-10-26. Sources fetched this session: Binoculars (arXiv HTML v3), Fast-DetectGPT (arXiv HTML v3), DIPPER/retrieval paper abstract, SynthID robustness study (arXiv 2508.20228), Google SynthID developer docs, Gemini forum threads, Raidar, Lastde, DetectRL and Glimpse abstracts, Groq and OpenRouter docs. Items marked "(not re-fetched)" are standard facts from well-known papers that I did not re-open this session. The report writer should treat their exact numbers as needing a check.

## Perplexity, log-rank/entropy baselines and GLTR

### Takeaway
The oldest detectors score text by how predictable each token is under a reference LM. Machine text has low perplexity, many tokens ranked top-1 or top-10, and low predictive entropy. These are cheap to compute but easy to break: they fail on unusual prompts (the "capybara problem"), on memorized human text, and once text is paraphrased.

### Cited Findings
- **Log-perplexity** (as Binoculars formalizes it): logPPL_M(s) = −(1/L) Σ_i log Y_{i,x_i}, the mean negative log-likelihood of the observed tokens under model M. Low values suggest machine text. — [Binoculars, Hans et al., arXiv 2401.12070v3](https://arxiv.org/html/2401.12070v3)
- **Log-rank, rank and entropy baselines.** Each token gets the rank (or log-rank) of the observed token in the model's sorted next-token distribution, or the entropy of that distribution, averaged over the text. Glimpse (ICLR 2025) lists Entropy, Rank, Log-Rank and Fast-DetectGPT as the standard "white-box" scorers that need full token distributions. — [Glimpse, arXiv 2412.11506](https://arxiv.org/html/2412.11506v2)
- **GLTR** (Gehrmann, Strobelt and Rush, ACL 2019 demo; uses GPT-2 117M) colors each token by the rank bucket of the observed token: top-10, top-100, top-1000 or beyond. Machine text is dominated by top-10 tokens, while human text has more top-100/1000+ tokens. In their human study, GLTR raised untrained readers' detection rate from about 54% to 72%. These results are from the GPT-2 era (2019). (not re-fetched) — [GLTR, arXiv 1906.04043](https://arxiv.org/abs/1906.04043)
- **Capybara problem.** An LLM's answer to an unusual prompt can have high perplexity, so raw perplexity misreads it as human. Binoculars' example is a GPT-4 passage about an astrophysicist capybara with a Falcon perplexity of 2.20. — [Binoculars](https://arxiv.org/html/2401.12070v3)
- **Memorized human text scores as machine.** The US Constitution got a Binoculars score of 0.76, on the machine side of the 0.901 threshold. 3 of 11 famous texts were on the machine side. — [Binoculars](https://arxiv.org/html/2401.12070v3)

### Inferences
- For a rewriting system, mean token log-prob and mean log-rank under a small LM are the most basic "machine-ness" signals to track. The fraction of tokens in GLTR's top-10 bucket is an easy-to-read version of the same thing.
- Pushing perplexity up indiscriminately, for example with random rare words, is a poor target. Very high-perplexity or random text is unnatural and is caught by ratio-based detectors in other ways (see Binoculars on random tokens below). The goal is a human-like distribution, not a maximum.

### Gaps
- I did not re-verify GLTR's 54% to 72% numbers this session.

## DetectGPT and Fast-DetectGPT (probability curvature)

### Takeaway
DetectGPT tests whether a text sits at a local maximum of log-probability, by comparing it with T5 perturbations. Fast-DetectGPT replaces perturbation with analytic sampling of alternative tokens from a sampling model at each position. The result is a z-score of how much more likely the observed tokens are than typical sampled alternatives. It is about 340x faster and about 75% relatively better. Both are badly hurt by strong paraphrasing (DetectGPT: 70.3% to 4.6% detection at 1% FPR under DIPPER).

### Cited Findings
- **DetectGPT** (Mitchell et al., ICML 2023). Perturbation discrepancy d(x) = log p_θ(x) − E_{x̃~q(·|x)} log p_θ(x̃), where q is T5 mask-filling. It is often normalized by the standard deviation of the perturbed log-probs. Machine text has strongly positive curvature, meaning most perturbations lower its probability. It needs about 100 perturbations, each scored by the model. (not re-fetched) — [DetectGPT, arXiv 2301.11305](https://arxiv.org/abs/2301.11305)
- **Fast-DetectGPT** (Bao et al., ICLR 2024; v1 8 Oct 2023). Conditional probability curvature d(x, p_θ, q_φ) = (log p_θ(x|x) − μ̃) / σ̃, where μ̃ = E_{x̃~q_φ(x̃|x)}[log p_θ(x̃|x)] and σ̃² is the variance. Here x̃ resamples each token independently, conditioned on the original prefix x_{<j}.
  - Analytic form: μ̃ = Σ_j μ̃_j and σ̃² = Σ_j (Σ_{x̃_j} q_φ(x̃_j|x_{<j}) log² p_θ(x̃_j|x_{<j}) − μ̃_j²). This needs only one forward pass of each model.
  - Black-box defaults are GPT-J as the sampling model and GPT-Neo-2.7B as the scoring model. The white-box setting uses the source model for both.
  - Source: [Fast-DetectGPT, arXiv 2310.05130v3](https://arxiv.org/html/2310.05130v3)
- **Fast-DetectGPT results** (Table 3, black-box, GPT-J/Neo-2.7; data generated with 2023 models):
  - ChatGPT: mean AUROC 0.9615 (XSum 0.9907, Writing 0.9916, PubMed 0.9021).
  - GPT-4: mean AUROC 0.9061 (PubMed 0.8503).
  - DetectGPT in the same table: 0.8223 (ChatGPT) and 0.6228 (GPT-4).
  - Speed: 233 s vs 79,113 s on XSum, about 340x faster.
  - Source: [Fast-DetectGPT](https://arxiv.org/html/2310.05130v3)
- The Fast-DetectGPT paper says it "consistently outperforms" zero-shot and trained detectors under paraphrasing, and it introduces a "decoherence attack". I could not read the appendix numbers (truncated). — [Fast-DetectGPT](https://arxiv.org/html/2310.05130v3)
- **DIPPER paraphrase attack** (Krishna et al., NeurIPS 2023; arXiv March 2023). An 11B T5-XXL paraphraser with controllable lexical and order diversity cut DetectGPT's detection accuracy from 70.3% to 4.6% at 1% FPR. It also evaded watermarking, GPTZero and OpenAI's classifier (the abstract gives no numbers for those). The proposed defense, retrieval over a database of the provider's own generations (15M generations), detected 80–97% of paraphrased generations at 1% FPR on human text. — [Krishna et al., arXiv 2303.13408](https://arxiv.org/abs/2303.13408)

### Inferences
- Fast-DetectGPT's score is essentially: how far is the observed log-prob above the expected log-prob of a token sampled from the model's own distribution, in standard-deviation units? Machine text picks tokens near the mode, so log p(x) is well above μ̃. Human text often picks tokens at or below the expected value.
- A rewriter wants the per-token gap (log p(x_j) − μ̃_j) to center near 0 with human-like spread. That is cheap to compute with one small model, using the same model as sampler and scorer (the white-box-style variant).
- Paraphrase results come from 2023 detectors and 2023 generators. Newer detectors such as Binoculars and Lastde claim better robustness but were not all tested against DIPPER-strength attacks.

### Gaps
- Fast-DetectGPT's appendix paraphrase numbers were not retrieved.
- There are no independent 2025–2026 numbers for Fast-DetectGPT against LLM-based humanizers.

## Binoculars (Hans et al., 2024)

### Takeaway
Binoculars divides the observer model's log-perplexity of the text by the cross-perplexity between two related models' next-token distributions. This normalizes away how surprising the prompt or topic is by itself. It reports more than 90% detection of ChatGPT text at 0.01% FPR with no training, but the paper explicitly does not evaluate adversarial paraphrasing.

### Cited Findings
- **Formula.** log X-PPL_{M1,M2}(s) = −(1/L) Σ_i M1(s)_i · log(M2(s)_i), the per-position cross-entropy between the two models' distributions.
  - Score: B = logPPL_{M1}(s) / log X-PPL_{M1,M2}(s). Lower means more machine-like.
  - The models must share a tokenizer. The defaults are Falcon-7B and Falcon-7B-Instruct, and the global threshold is about 0.901.
  - The paper is inconsistent about which model is "observer" and which is "performer".
  - Source: [Binoculars, arXiv 2401.12070v3](https://arxiv.org/html/2401.12070v3)
- **Why it works.** The denominator estimates the baseline surprise of the context, meaning how much one LLM is surprised by another LLM's next-token predictions. Dividing by it corrects for prompts that make even LLM text high-perplexity. — [Binoculars](https://arxiv.org/html/2401.12070v3)
- **Results** (v1 Jan 2024, v3 Oct 2024):
  - More than 90% of ChatGPT samples detected at 0.01% FPR.
  - Ghostbuster datasets, TPR at 0.01% FPR: Writing Prompts 0.93/0.95, News 0.95/0.99, Essays 0.98/0.99 (Falcon/LLaMA pairs).
  - Open Orca: 92% of GPT-3 and 89.57% of GPT-4 samples.
  - Gemini-1.0-pro was detected at 96.89%, but GPT-4 only at 41.86% in Table 8.
  - Source: [Binoculars](https://arxiv.org/html/2401.12070v3)
- **Robustness.** The paper does not evaluate paraphrase or evasion attacks. Changing the system prompt (for example to a "pirate" style) reduced sensitivity by only about 1%. Random token strings score about 1.35, above the human mean of about 1, so they read as human. Detection is weaker in non-English and low-resource languages. — [Binoculars](https://arxiv.org/html/2401.12070v3); [GitHub README](https://github.com/ahans30/Binoculars)
- The README warns against using it without human supervision, and says the fixed threshold was tuned for the Falcon pair. — [GitHub](https://github.com/ahans30/Binoculars)

### Inferences
- A cheap local Binoculars proxy is feasible with a base/instruct pair that shares a tokenizer, for example Qwen2.5-0.5B with Qwen2.5-0.5B-Instruct. The threshold would need recalibrating on known human and AI samples, since the 0.901 value is specific to Falcon.
- Binoculars mainly rewards text whose tokens are less predictable than the context's expected entropy suggests. Rewriting that adds idiosyncratic but plausible word choices raises B. Inserting noise also raises B, but it hurts quality.

### Gaps
- No primary-source paraphrase or humanizer robustness numbers were found for Binoculars.

## Ghostbuster, DNA-GPT, Raidar, RADAR and newer zero-shot methods

### Takeaway
Beyond single-model likelihood scores, detectors use:
- feature search over probabilities from several weak LMs (Ghostbuster),
- regeneration overlap (DNA-GPT),
- how much an LLM edits the text when asked to rewrite it (Raidar),
- adversarially trained classifiers (RADAR),
- time-series dynamics of token probabilities (Lastde),
- probability-distribution estimation from API top-k logprobs (Glimpse),
- geometric intrinsic dimension of embeddings (PHD).

Benchmarks such as DetectRL (NeurIPS 2024) find that even state-of-the-art detectors degrade under realistic edits.

### Cited Findings
- **Ghostbuster** (Verma et al., NAACL 2024). Text is passed through weaker LMs (unigram, trigram, GPT-3 ada/davinci) to get per-token probabilities. A structured search over vector and scalar combinations of these features is followed by a linear classifier. It reports about 99 F1 in-domain. (not re-fetched) — [Ghostbuster, arXiv 2305.15047](https://arxiv.org/abs/2305.15047)
- **DNA-GPT** (Yang et al., ICLR 2024). The text is truncated midway and the model regenerates continuations from the prefix. In black-box mode it compares n-gram overlap between the original suffix and the regenerations; in white-box mode it compares log-probability differences. Machine text overlaps more with its own regenerations. (not re-fetched) — [DNA-GPT, arXiv 2305.17359](https://arxiv.org/abs/2305.17359)
- **Raidar** (Mao et al., ICLR 2024). It prompts an LLM to rewrite the input and measures the edit distance. LLMs change human text more than AI text, because they rate AI text as higher quality. It improves detector F1 by up to 29 points across news, creative writing, essays, code, Yelp and arXiv. — [Raidar, arXiv 2401.12970](https://arxiv.org/abs/2401.12970)
- **RADAR** (Hu, Chen and Ho, NeurIPS 2023). A RoBERTa-style detector is trained adversarially against an LLM paraphraser, specifically to be robust to paraphrase. (not re-fetched) — [RADAR, arXiv 2307.03838](https://arxiv.org/abs/2307.03838)
- **Lastde / Lastde++** (Xu et al., Oct 2024). It treats the token probability sequence as a time series and combines local temporal-dynamics features with global statistics. It is training-free, and the authors claim state of the art on 6 datasets and greater robustness to paraphrasing than baselines (no numbers in the abstract). — [Lastde, arXiv 2410.06072](https://arxiv.org/abs/2410.06072)
- **Glimpse** (Bao et al., ICLR 2025). It estimates full next-token distributions from the partial top-k logprobs returned by proprietary APIs, so that Entropy, Rank, Log-Rank and Fast-DetectGPT can run on GPT-3.5-class scorers. It reports an average AUROC of about 0.95 across five recent source models, a 51% relative gain over open-source scorers. — [Glimpse, arXiv 2412.11506](https://arxiv.org/html/2412.11506v2)
- **DetectRL** (NeurIPS 2024 Datasets and Benchmarks). It covers real-world domains plus adversarial variants: varied prompts, human-like revisions such as word substitution, and writing noise such as typos. It finds that "even SOTA detection techniques still underperformed." — [DetectRL, arXiv 2410.23746](https://arxiv.org/abs/2410.23746)
- **Intrinsic dimension / PHD** (Tulchinskii et al., NeurIPS 2023). It estimates the persistent-homology dimension of a text's contextual token embeddings (RoBERTa). AI text has lower intrinsic dimension than human text, and the authors report the signal is fairly robust to paraphrase and stable across domains. (not re-fetched) — [arXiv 2306.04723](https://arxiv.org/abs/2306.04723)

### Inferences
- Raidar and DNA-GPT imply that output which a strong LLM would "leave alone" looks machine-like. A humanizer can test this cheaply: ask a small LLM to "improve" the text and measure the normalized edit distance. Higher means more human-like.
- These methods all measure, in different ways, how close the text is to an LLM's mode. Surface-level edits that keep mode-like word choices, such as synonym swaps of high-probability tokens, help less than changing the word-choice distribution itself.

### Gaps
- I did not verify exact numbers for Ghostbuster, DNA-GPT, RADAR or PHD this session.
- I found no consolidated 2025–2026 benchmark of these detectors against commercial humanizers.

## Watermarking: Kirchenbauer, SynthID-Text, OpenAI, and Gemini-based humanizers

### Takeaway
Green-list and tournament watermarks are statistical biases keyed on preceding tokens. They survive light edits but degrade under paraphrase and especially translation or back-translation.

The open question for a Gemini-based humanizer is whether the API itself watermarks. Google's developer docs do not say. A Google forum reply on 19 Aug 2026 says Gemini API and AI Studio text "IS SynthID-watermarked", which reverses the same poster's 5 Aug "NOT" answer. Treat this as unconfirmed but plausible. OpenAI built a text watermark (99.9% reliable on unedited text, per its Aug 2024 statement) but had not shipped it as of that statement.

### Cited Findings
- **Kirchenbauer et al.** (ICML 2023). A hash of the previous token(s) seeds a split of the vocabulary into a "green list" (fraction γ) and a red list, and δ is added to green-token logits before sampling.
  - Detection counts green tokens, |s|_G, over T tokens: z = (|s|_G − γT) / sqrt(T γ(1−γ)). For example, z > 4 means watermarked.
  - It needs no model access to detect, only the key and hash, and is detectable from roughly a few dozen tokens.
  - Paraphrase dilutes the green fraction.
  - (not re-fetched) — [arXiv 2301.10226](https://arxiv.org/abs/2301.10226)
- **SynthID-Text** (Dathathri et al., Nature, Oct 2024).
  - It modifies only the sampling procedure ("Tournament sampling" with pseudorandom g-functions seeded by recent context and a key), integrated with speculative sampling.
  - It was deployed in Gemini and Gemini Advanced. A live study of about 20 million responses found no significant difference in user ratings.
  - It was open-sourced in Hugging Face Transformers v4.46.0.
  - Sources: [HF blog](https://www.huggingface.co/blog/synthid-text); [summary of Nature deployment](https://aiweekly.co/alerts/deepminds-synthid-text-watermarks-live-gemini-responses); Nature article: https://www.nature.com/articles/s41586-024-08025-4 (paywall redirect, not read directly)
- **Google developer docs on SynthID Text.**
  - It is a logits processor applied after top-k/top-p, with keys (more keys means more layers), an n-gram length (default 5) and a sampling table of at least 2^16.
  - A Bayesian detector returns watermarked, not watermarked or uncertain.
  - It survives "changing a few words, or mild paraphrasing" but is "less effective on factual responses". Confidence "can drop substantially after thorough rewriting or translation", and it is "not designed to stop determined bad actors".
  - The page does not say whether Gemini outputs are watermarked, and it names no public text detection portal.
  - Source: [ai.google.dev SynthID](https://ai.google.dev/responsible/docs/safeguards/synthid)
- **Independent robustness study** (arXiv 2508.20228, Aug 2025; Sheared-LLaMA-1.3B generator, 200-token C4 samples, baseline F1 1.0):
  - DIPPER paraphrase (lex 10, order 5): F1 0.842 (TPR 0.895, FPR 0.23).
  - Back-translation via Chinese: F1 0.711 (TPR 0.675). French 0.738, Italian 0.813, Japanese 0.819.
  - Synonym substitution at 70%: F1 0.884.
  - Copy-paste dilution at 10x: F1 0.788.
  - These are small-model, open-source SynthID settings, not Gemini's production configuration.
  - Source: [arXiv 2508.20228](https://arxiv.org/html/2508.20228v1)
- A 2026 theoretical paper reportedly describes a "layer inflation attack" on SynthID's mean score. This is a search-snippet summary only and was not verified. — [search result](https://sotaverified.org/papers/260303410)
- **Gemini API watermarking status.**
  - Forum thread (Aug–Oct 2026) on EU AI Act Art. 50(2) compliance. A responder said on 5 Aug 2026 that API text is "NOT SynthID-watermarked". The same responder corrected this on 19 Aug 2026: API text "IS actually SynthID-watermarked", including AI Studio and antigravity.
  - The responder's role was not shown, there was no official doc link, and Vertex AI and the verification tool were left unanswered.
  - Public SynthID verification reportedly accepts only image, video and audio.
  - Source: [Google AI forum thread](https://discuss.ai.google.dev/t/does-gemini-api-text-output-carry-synthid-watermarking-gemini-2-5-flash-lite-gemini-3-1-flash-lite-eu-ai-act-art-50-2/177241)
- **OpenAI (Aug 2024 update to its provenance blog, after a WSJ report).** It has a text watermark that is 99.9% effective on unedited text. It is robust to localized edits and paraphrasing, but "trivial to circumvent" with global changes: translation, rewording with another generative model, or asking the model to insert then delete special characters. OpenAI also cited concerns about stigmatizing non-native speakers and had not deployed it at that time. — [Boing Boing summary](https://boingboing.net/2024/08/05/openai-could-watermark-the-text-chatgpt-generates-but-hasnt.html); [Thurrott](https://www.thurrott.com/?p=306664) (primary OpenAI post not fetched)
- Search results included a blog claiming "Claude now watermarks its text". The page failed to load and this is unverified. — [blopig](https://www.blopig.com/blog/?p=14726)

### Inferences
- If the humanizer uses Gemini to rewrite, its output may carry a SynthID watermark from the rewrite itself. Rewriting with Gemini does not remove a watermark; it may add one.
  - Mitigations would be a final pass through a non-watermarking model (for example a local open model), translation round-trips (which cost quality), or heavy non-LLM edits.
  - Since Google offers no public text-detection portal for third parties (per the forum and docs), the practical risk is mostly Google-side detection. No public verifier for Gemini's production key is known.
- Watermark evidence accumulates per token. Short outputs and low-entropy factual content carry weak watermarks, and mixing in human text (copy-paste dilution) lowers detection.

### Gaps
- There is no official Google documentation confirming SynthID on Gemini API text, its start date, or free-tier specifics.
- I could not read the Nature paper's own paraphrase numbers (paywall).

## Local proxy scoring and provider logprobs

### Takeaway
Fast-DetectGPT, the Binoculars-style ratio, mean log-rank, GLTR buckets and entropy can all be computed in one or two forward passes of a small open model on CPU. That makes a local proxy scorer realistic. Provider logprobs are unreliable: Groq rejects them, Gemini has removed them from 2.5 and 3.x models, and OpenRouter's support depends on the provider.

### Cited Findings
- Fast-DetectGPT's analytic form needs only the per-position distributions of the sampling and scoring models. It reached near-identical accuracy to sampling and was about 10% faster. The same model can serve as both sampler and scorer. — [Fast-DetectGPT](https://arxiv.org/html/2310.05130v3)
- Binoculars needs two models with a shared tokenizer. Its threshold depends on the model pair. — [Binoculars](https://arxiv.org/html/2401.12070v3); [GitHub](https://github.com/ahans30/Binoculars)
- Glimpse shows that top-k logprobs from an API (k ≤ 20) can be extrapolated to approximate full distributions for these scores. — [Glimpse](https://arxiv.org/html/2412.11506v2)
- **Groq:** its OpenAI-compatibility docs list logprobs and top_logprobs as unsupported, returning a 400 error. — [Groq docs](https://console.groq.com/docs/openai); [Groq community request](https://community.groq.com/t/add-support-for-logprobs-in-model-api-response/193)
- **OpenRouter:** it accepts `logprobs` (bool) and `top_logprobs` (0–20), but support depends on the upstream provider. Users are told to check provider pages or require parameter support in routing. — [OpenRouter parameters](https://openrouter.ai/docs/api/reference/parameters)
- **Gemini:**
  - The July 2025 Google blog documents `responseLogprobs` and `logprobs` (1–20) on Vertex AI. — [Google Developers Blog](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/)
  - Forum reports say 2.5-flash started returning "Logprobs is not enabled" 400s. — [forum](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989)
  - Logprobs are not returned for Gemini 3.x on Vertex or AI Studio, which a Google-affiliated responder called intended. — [forum](https://discuss.ai.google.dev/t/missing-logprobs-support-in-the-newest-gemini-models-3-1-pro-3-6-flash-on-vertex-ai-and-ai-studio/176557)
- Note that API logprobs are only for generated tokens. Scoring an arbitrary input text needs prompt logprobs (an "echo" mode), which none of these docs advertise. — inference from the parameter descriptions above ([OpenRouter](https://openrouter.ai/docs/api/reference/parameters): "log probabilities of the output tokens")

### Inferences
- **Recommended cheap proxy stack** (one forward pass of Qwen2.5-0.5B or GPT-2 via transformers or llama.cpp on CPU, on the order of 1 second per few hundred tokens; not benchmarked here):
  1. Mean token log-prob, i.e. log-perplexity, plus its per-sentence variance.
  2. Mean log-rank and the GLTR bucket fractions (top-1, top-10, top-100, and beyond).
  3. Mean entropy.
  4. Fast-DetectGPT curvature (analytic, same model as sampler and scorer).
  5. Optionally, a Binoculars ratio using the base and instruct variants of the same family.
- Calibrate thresholds on a small local set of known human text and the system's own raw LLM output. Track z-scores relative to that human reference, not absolute values.
- A Raidar-style "rewrite distance" check can use any cheap chat API, since it needs no logprobs.

### Gaps
- No measured CPU latency or accuracy figures for 0.5B-scale proxies were found. I also found no evidence on how well small-model scores correlate with commercial detectors such as GPTZero, Turnitin or Originality.

## Sentence-length burstiness and surface statistics

### Takeaway
Burstiness, the variance of sentence length or perplexity across sentences, is a popular heuristic (notably GPTZero's marketing). The evidence that it matters to modern token-probability detectors is weak. One unverified vendor analysis finds sentence-length burstiness near chance (AUROC 0.521) on matched pairs, and a 2026 NeurIPS submission argues it is confounded with length. The academic detectors above do not use sentence-length burstiness at all.

### Cited Findings
- Vendor analysis: sentence-length burstiness had an AUROC of 0.521 on 5,935 register- and length-matched pairs. Machine text was slightly *more* variable in sentence length. Not peer-reviewed. — [Opace](https://opace.agency/tools/ai/content-verification-integrity/research/ai-writing-myths-measured/)
- The NeurIPS 2026 submission "Burstiness Was Measured Wrong, and Prompting Cannot Aim It" argues the common operationalization is an unreviewed heuristic confounded with length (raw variance grows with the mean sentence length). It covers 9 model families on 200 matched Wikinews articles; results were not visible in the abstract. — [NeurIPS 2026 virtual page](https://neurips.cc/virtual/2026/164820)
- None of the formulas above (Binoculars, Fast-DetectGPT, DetectGPT, perplexity, rank, entropy) contains sentence-length terms. All operate on token probabilities. — [Binoculars](https://arxiv.org/html/2401.12070v3); [Fast-DetectGPT](https://arxiv.org/html/2310.05130v3)
- Binoculars was largely insensitive to the style of the generating prompt; the "pirate" system prompt cut sensitivity by only about 1%. This suggests surface stylistic changes do not move token-statistic detectors much. — [Binoculars](https://arxiv.org/html/2401.12070v3)

### Inferences
- Varying sentence length is cheap and may help against commercial classifiers that include it as a feature. However, the token-level word-choice distribution (rank and log-prob gap) is what the zero-shot literature measures, so that should be the primary target.
- If burstiness is tracked, normalize it, for example with the coefficient of variation of sentence lengths, and compare it with human reference text in the same register.

### Gaps
- No primary peer-reviewed quantification of burstiness's weight in commercial detectors such as GPTZero or Turnitin was found.
