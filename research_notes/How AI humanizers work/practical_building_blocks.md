# Practical, Low-Cost Building Blocks for an LLM Humanizer (Style Rewriter) on Free/Cheap APIs, plus Responsible Positioning

Research date: 2026-10-26. 18 tool calls. Numbers carry the dates of their sources. Items marked "Inference" are my reasoning and are not sourced.

## 1. Style transfer and imitating an author's own style (few-shot exemplars, descriptors, "voice cloning")

### Takeaway
Few-shot exemplars of an author's real writing work better than zero-shot prompting, but the benefit levels off after 2 examples. LLMs still fail to imitate informal, everyday styles such as blogs and forums. The strongest recent evidence (Epoch AI, July 2026) shows that 5-shot style imitation lets about 10-18% of passages evade top commercial detectors, against under 1% for generic prompts. Exemplars of the user's own writing are therefore a real but modest lever, and they work best for formal or scientific genres.

### Cited Findings
- "Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors" (Wang, Tripto, Park, Li, Zhou; Findings of EMNLP 2025). It evaluated over 40,000 generations per model on 400+ real authors across news, email, forums and blogs. The metrics combined authorship attribution, authorship verification, style matching and AI detection. Few-shot beat zero-shot on every metric. Moving from 2 to 10 examples added little, so in-context learning saturates. — [arXiv 2509.14543](https://arxiv.org/abs/2509.14543v1)
- In the same study, imitation was reasonably good for structured, formal domains (news CCAT50, Enron email) and weak for informal ones (Reddit, blogs). One summary reports authorship-verification accuracy of about 17-21% for blogs at 5-shot. This figure comes from a third-party review, so check it against the paper before relying on it. — [Moonlight review](https://www.themoonlight.io/en/review/catch-me-if-you-can-not-yet-llms-still-struggle-to-imitate-the-implicit-writing-styles-of-everyday-authors); [arXiv](https://arxiv.org/abs/2509.14543v1)
- The authors conclude that closing the gap probably needs richer personalization signals or hybrid prompting plus fine-tuning. — [arXiv 2509.14543](https://arxiv.org/abs/2509.14543v1)
- Epoch AI (published 15 Jul 2026) used 495 human passages of about 500 words from 99 authors (blog, fiction, scientific). For each author, Claude Opus 4.8, GPT-5.5 and Gemini 3.1 Pro wrote two passages: one "style imitation" passage after seeing 5 of the author's real passages, and one from a basic generic prompt. Detectors were Pangram 3.3.2, GPTZero (2026-05-11-base) and Originality.ai Turbo 3.0.2. — [Epoch AI](https://epoch.ai/data-insights/ai-detectors-false-negatives)
  - Basic-prompt AI was missed almost never: Pangram 0/297, GPTZero 2/297 (0.7%), Originality 1/297 (0.3%).
  - 5-shot style-imitation AI was missed far more often: Pangram 30/297 (10.1%), GPTZero 32/297 (10.8%), Originality 53/297 (17.8%), about 13% on average.
  - Genre mattered most. Scientific imitation was missed about 24-29%, against blog 4-19% and fiction 1-5%. The most-evaded cells were Gemini-scientific for Pangram (48%) and GPTZero (36%).
  - Human false positives: Pangram 0/495, GPTZero 0/495, Originality 19/495 (3.8%).
  - The study did not test paraphrasers or humanizers. Its only evasion method was 5-shot imitation.
- The DetectLLM appendix shows that detection by zero-shot, perturbation-free methods (log-likelihood, log-rank) improves as generation temperature decreases. — [DetectLLM arXiv 2306.05540](https://arxiv.org/pdf/2306.05540)

### Inferences
- The app could add a "My writing samples" box: 2-3 short samples of the user's own prose, about 300-600 tokens in total, sent with each rewrite request. The literature suggests 2 samples capture most of the gain, so more exemplars mostly cost tokens. To save tokens, extract a compact style card once, by LLM or locally (sentence-length distribution, contractions, punctuation habits, favourite connectives, person and voice), and cache it. Then send the card plus one short exemplar instead of several full samples.
- Epoch's genre result suggests rewrites of formal or technical text may gain more from exemplars than casual text does, which matches the EMNLP finding that formal styles are easier to imitate.
- The Epoch numbers apply to frontier models. Free models such as gpt-oss-120b and Qwen3 will probably imitate less well. This is unmeasured.

### Gaps
- No source found that measures style descriptors (a text "style card") against raw exemplars head-to-head, or reports per-token efficiency.
- No study found on style imitation with free or open models (gpt-oss, Qwen3, Llama) against current detectors.

## 2. Sampling settings (temperature, top-p, min-p, typical) and whether free APIs expose them

### Takeaway
Changes in sampling strategy and repetition penalties are among the factors that measurably degrade detectors (RAID, ACL 2024). Low temperature makes text easier for likelihood-based detectors. Min-p keeps text coherent at higher temperatures. OpenRouter exposes the richest set of knobs. Groq offers only temperature and top_p, with no logprobs and n=1. Gemini offers temperature, topP, topK and candidateCount, and logprobs only on some models.

### Cited Findings
- RAID (Dugan et al., ACL 2024) contains more than 6M generations from 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. It tested 12 detectors (8 open-source, 4 closed) and found them "easily fooled" by adversarial attacks, shifts in sampling strategy, repetition penalties and unseen generators. — [RAID arXiv 2405.07940](https://arxiv.org/pdf/2405.07940); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10563509)
- DetectLLM reports that all perturbation-free zero-shot detectors performed better as temperature decreased. Text from Neo-2.7 at T=0.5 was detected almost perfectly by log-likelihood and log-rank on WritingPrompts. — [arXiv 2306.05540](https://arxiv.org/pdf/2306.05540)
- An October 2025 paper reports that state-of-the-art detectors are highly sensitive to generation parameters, with AUROC dropping in extreme cases from 0.99 to 0.01. This was seen in a search snippet only and not verified in full text. — [arXiv 2510.13681](https://arxiv.org/pdf/2510.13681v1)
- Min-p sampling (Nguyen et al., ICLR 2025 oral) truncates at a threshold that scales with the top token's probability. It beats top-p, top-k and η-sampling at keeping text coherent while allowing more diversity, especially at high temperature. Human evaluation preferred min-p, and Hugging Face and vLLM have adopted it. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/afa5f124e36bed5cc2125067005d43f5-Abstract-Conference.html); [arXiv 2407.01082](https://arxiv.org/abs/2407.01082v1)
- **OpenRouter** documents `temperature` (0-2, default 1.0), `top_p` (0-1), `top_k` (default 0, disabled), `min_p` (0-1), `top_a` (0-1), `repetition_penalty` (0-2), `logprobs` (bool) and `top_logprobs` (0-20). Support varies by provider: "refer to the model's provider section to confirm which parameters are supported." Omitted parameters fall back to the provider's defaults. — [OpenRouter parameters](https://openrouter.ai/docs/api/reference/parameters)
- **Groq**: `logprobs`, `logit_bias`, `top_logprobs` and `messages[].name` are "not supported and will result in a 400 error." `n` must equal 1. temperature=0 is converted to 1e-8, and valid values are greater than 0 and at most 2. — [Groq OpenAI compatibility](https://console.groq.com/docs/openai)
- **Groq** supports top_p (0-1) according to third-party wrapper docs. I found no Groq documentation on min_p. — [Nexlm Groq provider docs](https://hexdocs.pm/nexlm/Nexlm.Providers.Groq.html)
- **Gemini API**: the GenerateContent reference shows `candidateCount`, `temperature`, `topP`, `topK`, `maxOutputTokens` and `stopSequences` in its examples. The response schema includes `avgLogprobs` per candidate and a `LogprobsResult` with `topCandidates` per decoding step. — [Gemini API generate-content](https://ai.google.dev/api/generate-content)

### Inferences
- A practical default is temperature of about 0.9-1.1 with min_p of about 0.05-0.1 where OpenRouter routes to a provider that supports min_p, and temperature of about 1.0 with top_p of about 0.95 on Groq. Avoid T≤0.5 for the rewrite step, because low temperature makes likelihood detectors' job easier (DetectLLM). The exact values are untested heuristics, so the app should A/B them with its own detector-rescoring loop.
- Keep the detection and judging calls at low temperature for consistency, and use higher temperature only for the rewrite.
- Because Groq requires n=1, best-of-N on Groq costs N separate requests. Gemini's `candidateCount` may return several candidates in one call and share the input-token cost, which is useful for best-of-N if the free tier allows it (not verified).

### Gaps
- No primary evidence found that measures typical sampling or min-p specifically against modern detectors.
- I could not confirm from Groq's own docs the full parameter list for gpt-oss-120b or Qwen3, including min_p and top_k.
- Gemini's free-tier limits on candidateCount are unverified.

## 3. Best-of-N with a cheap local scorer: open detectors and small-model perplexity/Binoculars on CPU

### Takeaway
Several local scorers are feasible. A 0.4B DeBERTa-v3-large detector (desklib, MIT licence, claimed RAID leader), an older RoBERTa-base HC3 detector, and Binoculars or Fast-DetectGPT with swapped-in small models can all run on CPU. Binoculars' published accuracy (>90% TPR at 0.01% FPR) depends on 7B Falcon models, and accuracy with sub-2B models is not documented. Treat any local score as a relative ranking signal for best-of-N, not as a calibrated "AI probability."

### Cited Findings
- **Binoculars** (Hans et al., 2024) is zero-shot and contrasts two closely related LLMs. It "detects over 90% of generated samples from ChatGPT (and other LLMs) at a false positive rate of 0.01%" without training on ChatGPT data. — [arXiv 2401.12070](https://arxiv.org/abs/2401.12070)
- Binoculars repo details:
  - Its fixed threshold was selected with Falcon-7B and Falcon-7B-Instruct, and other scoring models can be passed to the class.
  - Licence is BSD-3-Clause, and the code is described as for academic use with human supervision advised.
  - Detection is more reliable for English.
  - The README gives no CPU or small-model guidance and no memory figures.
  — [GitHub ahans30/Binoculars](https://github.com/ahans30/Binoculars)
- **Fast-DetectGPT** uses gpt-neo-2.7B as its default sampling and scoring model, and its README suggests gpt-j-6B for better accuracy. Its headline AUROC is about 0.9887. On a different benchmark (SpecDetect, XSum, T=0.8, top-p 0.96, top-k 40), it scored AUC 0.77 on Neo-2.7 text. — [search snippets of the repo summary / SpecDetect arXiv 2508.11343](https://arxiv.org/pdf/2508.11343)
- **desklib/ai-text-detector-v1.01** is a fine-tune of microsoft/deberta-v3-large with 0.4B parameters in F32 and an MIT licence. Its model card says it "currently leads the RAID Benchmark" (no figure or date on the card). The example code truncates input at 768 tokens. — [Hugging Face model card](https://huggingface.co/desklib/ai-text-detector-v1.01)
- **Hello-SimpleAI/chatgpt-detector-roberta** is RoBERTa-base trained for 1 epoch on all HC3 answers (full texts and split sentences, no held-out split). It is tied to the Jan 2023 paper arXiv 2301.07597. The card gives no licence and no accuracy. — [Hugging Face model card](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta)
- RAID found that detectors generalize poorly to unseen generators and domains and are fooled by repetition penalties and sampling changes. A local detector trained on older ChatGPT data (HC3) is therefore likely to be miscalibrated on gpt-oss or Qwen3 output. — [RAID](https://arxiv.org/pdf/2405.07940)

### Inferences
- **Recommended lightweight stack.** These figures are estimates from parameter counts, not measured benchmarks:
  - desklib DeBERTa-v3-large (0.4B): about 1.6 GB of F32 weights, or about 0.4-0.8 GB if quantized with ONNX int8. One 512-token pass on a modern 4-8-core CPU should take a few hundred ms to about 1-2 s.
  - GPT-2 small (124M) or Qwen2.5-0.5B perplexity: about 0.5-2 GB RAM, sub-second per paragraph.
  - Binoculars with a Qwen2.5-0.5B / 0.5B-Instruct pair (same tokenizer, as Binoculars requires): two forward passes, roughly 2-3 GB RAM.
- At 2,500 words (about 3,300 tokens), scoring N=3 candidates per flagged passage is CPU-cheap compared with the LLM tokens spent, so best-of-N moves the cost from evaluation tokens to N× generation tokens. On Groq that means N requests, because n=1.
- **Cheaper alternative.** Generate a single output that contains 2-3 alternative rewrites of a short passage, rank them locally with the zero-token checks the app already has plus one local model, and keep the best. This avoids paying N× for the shared prompt.
- Use the local score only for ranking within a passage, never as a user-facing probability. Calibrating against the app's own held-out human and AI samples is advisable.

### Gaps
- No published accuracy found for Binoculars with sub-2B observer/performer pairs such as Qwen2.5-0.5B or SmolLM.
- No measured CPU latency or RAM figures found for desklib, Binoculars or Fast-DetectGPT. The figures above are my estimates.
- desklib's actual RAID score and its submission date were not on the model card.
- TMR detector not researched (out of call budget).

## 4. Token logprobs from free APIs (computing candidate perplexity without a local model)

### Takeaway
Do not rely on API logprobs. Groq rejects them (HTTP 400). OpenRouter exposes them only for some providers. On Gemini, support has been withdrawn model by model: Google staff say Gemini 3.x no longer returns logprobs, and 2.5-flash reportedly lost them, with a separate quota on logprobs requests. A small local model is the dependable route to perplexity.

### Cited Findings
- Groq lists `logprobs` and `top_logprobs` as unsupported, and supplying them causes a 400 error. — [Groq docs](https://console.groq.com/docs/openai)
- OpenRouter accepts `logprobs` and `top_logprobs` (0-20), but provider support varies per model. — [OpenRouter parameters](https://openrouter.ai/docs/api/reference/parameters)
- The Gemini API response schema includes `avgLogprobs` and `LogprobsResult.topCandidates`. — [Gemini API reference](https://ai.google.dev/api/generate-content)
- Gemini forum reports:
  - `response_logprobs` returned 400 on gemini-1.5-pro-002, 1.5-flash-002, 2.0-flash-exp and exp-1206. One user hit a 429 "daily limit for requests with logprobs." — [Google AI forum](https://discuss.ai.google.dev/t/get-logprobs-at-output-token-level/54418)
  - Logprobs reportedly stopped working on gemini-2.5-flash ("not enabled") while 2.0-flash and flash-lite still worked. — [Google AI forum](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989)
  - A Google staff member said logprobs are "no longer returned for 3.X models" and that this is working as intended. — [Google AI forum](https://discuss.ai.google.dev/t/were-logprobs-disabled-for-gemini-3-3-1-in-vertex-api/132426); [also](https://discuss.ai.google.dev/t/missing-logprobs-support-in-the-newest-gemini-models-3-1-pro-3-6-flash-on-vertex-ai-and-ai-studio/176557)

### Inferences
- Even where API logprobs exist, they are the generating model's own likelihood of its own sampled output, not the likelihood an independent observer would assign. That is a weak proxy for how detectors see the text. A fixed local observer model gives comparable scores across providers.

### Gaps
- No authoritative per-model list found of which OpenRouter free models return logprobs.
- Forum evidence on Gemini is anecdotal, apart from the staff statement on 3.x.

## 5. Edit-based vs full regeneration; meaning and fact preservation checks

### Takeaway
Full paraphrasing is very effective against detectors: DIPPER cut DetectGPT from 70.3% to 4.6% detection at 1% FPR without appreciably changing meaning. But the detection literature also shows that even minimal AI polishing of human text gets flagged. So "minimal edits" to AI text probably keep most of the AI signal, and edits to human text can add AI signal. Cheap meaning checks are feasible locally: all-MiniLM-L6-v2 has 22.7M parameters and an Apache-2.0 licence. Combine it with the existing number and entity checks.

### Cited Findings
- DIPPER (Krishna et al., 2023) is an 11B paraphraser that evaded watermarking, GPTZero, DetectGPT and OpenAI's classifier. "DIPPER drops detection accuracy of DetectGPT from 70.3% to 4.6% (at a constant false positive rate of 1%)" without appreciably changing meaning. Their retrieval-based defense, a provider-side database of 15M generations, detected 80-97% of paraphrased generations at about 1% human false positives. — [arXiv 2303.13408](https://arxiv.org/abs/2303.13408)
- RAID: adversarial editing attacks cause large drops in detector performance. — [RAID](https://par.nsf.gov/servlets/purl/10563509)
- "Almost AI, Almost Human" (Saha & Feizi, ACL 2025) tested 12 detectors on APT-Eval, 14.7K human texts polished by LLMs at varying intensities. Detectors "frequently flag even minimally polished text as AI-generated," cannot tell degrees of AI involvement apart, and are biased against text from older and smaller models. — [arXiv 2502.15666](https://arxiv.org/abs/2502.15666)
- all-MiniLM-L6-v2 has 22.7M parameters, 384-dimensional embeddings and an Apache-2.0 licence. It truncates at 256 word pieces and was trained at 128 tokens. ONNX and OpenVINO variants are listed. — [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Inferences
- The two findings pull in opposite directions. Light edits are cheap and keep meaning, but detectors pick up even small AI-made changes. Sentence-level restructuring (DIPPER-style lexical and order diversity) moves scores but risks drift in meaning.
- A pragmatic design is to regenerate only the flagged sentences, with a per-passage cap on change (for example, a token-level edit ratio).
- Gate every rewrite on three checks:
  1. MiniLM cosine against the source passage, with a threshold tuned locally (roughly 0.8-0.85 is a common starting point, not sourced).
  2. Exact match of numbers, dates and named entities. The app already does this; adding spaCy small NER on CPU is cheap.
  3. A negation and modal check ("not", "may", "must"), because embeddings are weak at catching polarity flips.
- MiniLM's 256-word-piece limit means scoring sentence by sentence or passage by passage, which fits the app's per-passage design. On CPU it should take milliseconds per sentence (estimate).
- An LLM-as-judge for meaning costs tokens. Reserve it for cases where the cosine or entity check is borderline.

### Gaps
- No study found that directly compares minimal-edit rewriting with full regeneration on both detector score and semantic similarity using current free models.

## 6. Grounding: asking the writer for specifics (content injection) and process-transparency tools

### Takeaway
I found no rigorous study showing that adding personal or specific author content lowers detector scores. The closest evidence is indirect: genuine human-written text is rarely flagged by top detectors (0/495 for Pangram and GPTZero, July 2026). The established product pattern in this area is process transparency (Grammarly Authorship), not output disguise.

### Cited Findings
- Grammarly Authorship sorts text typed into a document as human-typed, AI-generated, AI-modified, pasted (from a known or unknown source) or edited by Grammarly or a spell checker. It gives instructors a replay of the writing process, runs in Google Docs and Word, and integrates with Canvas and Blackboard. — [Grammarly Authorship](https://www.grammarly.com/edu/authorship); [Grammarly blog](https://www.grammarly.com/blog/institutions/grammarly-authorship-gives-faculty-new-insights-into-the-student-ai-writing-process/)
- Grammarly claims institutions can cut integrity violations by 96% or more, citing Rowan-Cabarrus Community College. This is vendor marketing and not independently verified. — [Grammarly edu outcomes](https://www.grammarly.com/edu/outcomes)
- Grammarly's guidance positions AI detection as part of "a holistic approach to evaluating writing originality." — [Grammarly blog](https://www.grammarly.com/blog/ai-detector-authorship/)
- Human false-positive rates on 495 pre-2022 passages: Pangram 0%, GPTZero 0%, Originality.ai 3.8%. — [Epoch AI, Jul 2026](https://epoch.ai/data-insights/ai-detectors-false-negatives)

### Inferences
- An "ask the writer" step serves both quality and honesty. The app would prompt the user for 1-3 concrete details per flagged passage (an example, a number, a personal observation, a local reference) and splice in only what the user typed. That avoids fabrication, which is a hard constraint for this app.
- Text the user actually wrote is, by Epoch's data, the content top detectors are least likely to flag. More importantly, it makes the document the user's own work. The UI could also log which spans were user-typed versus model-rewritten, a lightweight local "authorship trail" in the spirit of Grammarly Authorship.

### Gaps
- No controlled study found that measures detector scores before and after injecting author-supplied specifics into AI drafts.
- No products found, other than process-tracking tools, that use an interactive "ask the writer for specifics" flow.

## 7. Ethics and positioning: academic integrity, detectors, vendor terms, responsible framing

### Takeaway
The environment has turned against evasion-branded tools. Turnitin added "AI bypasser" (humanizer) detection on 27 Aug 2025 and reportedly tracks about 150 humanizers. OpenAI's usage policies prohibit using its services for academic dishonesty. Universities are moving away from detector-only enforcement. A defensible positioning is a style and clarity editor that preserves facts, encourages the author's own voice and specifics, and does not claim to beat detectors.

### Cited Findings
- Turnitin announced on 27 Aug 2025 that its AI writing detection can flag text "intentionally modified by AI humanizer tools." The feature is built into AI writing detection (Originality add-on, iThenticate 2.0) and is English-only. Turnitin says it trained on "signals and patterns of leading humanizers." Plagiarism Today notes that independent testing is still needed. — [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/); [Campus Technology](https://campustechnology.com/articles/2025/09/02/new-turnitin-detection-feature-helps-identify-use-of-ai-humanizer-tools.aspx)
- Pangram says it flags "humanized" text, and Turnitin reportedly tracks about 150 humanizer tools. This is from a news alert, a secondary source. — [AI Weekly](https://aiweekly.co/alerts/pangram-and-grammarlys-authorship-fuel-a-campus-ai-arms-race)
- Yale, Vanderbilt, Johns Hopkins and Indiana University restrict or discourage using AI detectors as the sole evidence. Students at Yale, Adelphi and the University of Minnesota have sued over wrongful accusations. This is also secondary. — [AI Weekly](https://aiweekly.co/alerts/pangram-and-grammarlys-authorship-fuel-a-campus-ai-arms-race)
- Detectors flag even minimally AI-polished human text, which raises the risk of false accusations. — [arXiv 2502.15666](https://arxiv.org/abs/2502.15666)
- OpenAI's usage policies (unified 29 Oct 2025) prohibit using its services for "academic dishonesty," alongside deceit and fraud. This matters to the app if it routes through gpt-oss on OpenAI-adjacent terms, and in any case it reflects industry norms. — [OpenAI usage policies](https://openai.com/policies/usage-policies/)
- When students ask Grammarly's AI to compose text, it redirects them to their institution's academic integrity policy, and institutions control whether generative features are enabled. — [Grammarly edu](https://www.grammarly.com/edu)

### Inferences
- **Product guidance:**
  1. Frame the product as "make AI-assisted drafts sound like you: vary rhythm, cut clichés, keep facts." Do not make "bypass Turnitin/GPTZero" claims, and avoid "undetectable" marketing.
  2. Show the detector score as a style diagnostic with an explicit caveat that detectors are unreliable and not proof of authorship.
  3. Add an academic-use notice that tells students to follow their institution's AI policy and disclose AI assistance.
  4. Offer a disclosure export, such as an "AI-assisted; rewritten for style" note or a change log of user-typed versus model-rewritten spans.
  5. Prioritize the "ask the writer for specifics" step, which improves quality and authenticity together.
- Review the terms of each free provider (OpenRouter upstream providers, Groq, Gemini) for prohibited-use clauses. Only OpenAI's was confirmed here.
- Turnitin's humanizer detection means the app's "median 50% likely AI" after rewriting may not carry over to Turnitin. That is another reason not to market evasion.

### Gaps
- I did not retrieve Groq, OpenRouter or Google Gemini API prohibited-use terms for academic dishonesty.
- No independent accuracy evaluation found for Turnitin's bypasser detection.
- No primary university guidance page specifically about humanizer tools was retrieved. The university items above come from a secondary news source.
