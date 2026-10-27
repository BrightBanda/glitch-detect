# Commercial AI-Text Detectors and How They Catch Humanized / Paraphrased / Bypasser Text (2024-2026)

Notes compiled 2026-10-26. "VENDOR-REPORTED" marks numbers published by the detector's own company. Independent numbers are labelled with the study.

## Turnitin: models (AIW-1/AIW-2/AIR-1), AI-paraphrase detection, 2025 "AI bypasser" detection, scoring, thresholds, FP policy

### Takeaway
Turnitin runs a transformer classifier (AIW-2, Dec 2023) over sliding windows of a few hundred words, averages the window scores per sentence, and labels a document "AI" only if 20% or more of its sentences pass a threshold. A second model (AIR-1, July 2024) then tags AI sentences as "AI-paraphrased." In August 2025 it added "AI bypasser" detection, trained on the outputs of leading humanizers. All accuracy figures are vendor-reported. AIW-2's recall on AI-paraphrased text (78% at document level) is clearly lower than its standard recall (91%), and no number has been published for bypasser recall.

### Cited Findings
**Model history and architecture (Turnitin white paper, updated Aug 2024, VENDOR):**
- AIW-1 launched April 2023 and had processed more than 250M submissions by June 2024. AIW-2 replaced it in December 2023, with a stated focus on detecting "AI generated text that has been modified by AI paraphrasing tools ('text spinners')". AIR-1 ("AI rewriting detection") launched July 2024 to power AI-paraphrase highlighting — [Turnitin white paper (Aug 2024), hosted by Univ. at Buffalo](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)
- Both models are built on "a state-of-the-art transformer deep-learning architecture". Turnitin says it chose this over "a simpler model that relies primarily on hand-curated measures such as perplexity and burstiness that do not capture many higher order deviations" — [same white paper](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)
- **Segment and sentence scoring:** the model reads a window of "roughly a few hundred words (about five to ten sentences)" and moves it forward one sentence at a time. Each window gets a score from 0 to 1. A sentence's score is a weighted average of the windows that contain it, compared against a sentence-level threshold tuned to maximize recall while keeping FPR low. A document is labelled "AI-generated" if more than 20% of its sentences exceed that threshold. The 20% cutoff and the threshold were chosen "to keep document-level FPR below 0.01 (1%)". Minimum length is 300 words — [same](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)
- **Two-stage pipeline:** AIR-1 runs only when AIW-2 predicts 20% or more AI, and only on sentences AIW-2 already marked as AI. Sentences that are both AI-generated and AI-paraphrased are highlighted purple. AIR-1 can never turn a "human" sentence into an "AI" one — [same](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)
- **Training data / anti-paraphrase strategy:** the key change from AIW-1 to AIW-2 was adding "AI+AI paraphrased" text to training. The AI side was generated to "mirror text known to be human-written". The training set covered prompt strategies from "write the whole essay for me" to mixed human/AI text, and included "human-written+AI-paraphrased" and "AI-generated+AI-paraphrased" text. Turnitin says it over-sampled second-language writers and less common subjects. AIR-1's rationale: paraphrasing models "use a different generative architecture framework from a standard LLM, and therefore leave a distinct and detectable signature." Turnitin says AIW-2 was "still predominantly trained to detect writing outputs from GPT-3.5" — [same](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)
- **Numbers (VENDOR, Aug 2024):**
  - On 719,877 pre-2019 student papers: document-level FPR 0.51% for AIW-2 vs 0.70% for AIW-1; sentence-level FPR 0.33% vs 0.42%.
  - Document recall on 2,970 held-out docs: 91.18% for AIW-2 vs 89.83% for AIW-1.
  - On 1,768 AI-generated then AI-paraphrased documents: document recall 78.34% for AIW-2 vs 51.7% for AIW-1; sentence recall 85.22% vs 45.56%.
  - Recall by generator: GPT-4 76.8% (AIW-1: 39.7%), Llama-2 99.2%, Gemini Pro 1.0 87.25%.
  - ESL bias test on about 9,000 docs (ASAP/ICNALE/PELIC): L2 FPR 0.86% vs L1 FPR 0.87%, with no statistically significant difference.
  - AIW-2+AIR-1 sentence-level results: for AI+paraphrased sentences, 81.68% correctly tagged, 9.31% called human, 9.01% called AI-only. For AI-only sentences, 86.25% correct, 10.84% called human, 2.91% wrongly called paraphrased ("Type 1", which Turnitin prioritized minimizing).
  — [same](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)
- Non-English detectors use the same architecture with a new base model and tokenizer. There is no non-English AIR-1. A number is displayed only when more than 20% of qualifying text is predicted AI — [same, addendum](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)

**Display thresholds and false-positive policy:**
- Since July 2024, scores from 1% to 19% are shown as "*%" with no number and no highlights, because Turnitin considers that range less reliable. This applies to new submissions only — [BUE library guide](https://bue.libguides.com/c.php?g=144329&p=9698569) (secondary; I could not reach the primary Turnitin guide, which returned 403 / Cloudflare)
- In 2023 Turnitin's CPO put the sentence-level FPR at about 4%, alongside the document-level claim of under 1% for documents with 20% or more AI — [Inside Higher Ed (June 2023)](https://www.insidehighered.com/news/quick-takes/2023/06/01/turnitins-ai-detector-higher-expected-false-positives); [Turnitin blog](https://www.turnitin.com/blog/understanding-the-false-positive-rate-for-sentences-of-our-ai-writing-detection-capability). This conflicts with the 0.33% sentence FPR in the 2024 white paper, which was measured on a different model (AIW-2) and dataset.

**AI bypasser detection (Aug 2025):**
- Announced about August 27, 2025. Turnitin says it is part of AI writing detection, not a separate product. It is available in Turnitin Originality and the iThenticate 2.0 AI add-on, and trained and tested only on English — [AAP/PR Newswire release](https://aapnews.aap.com.au/news/cision20250827AE59725); [Campus Technology, Sept 2, 2025](https://campustechnology.com/Articles/2025/09/02/New-Turnitin-Detection-Feature-Helps-Identify-Use-of-AI-Humanizer-Tools.aspx)
- CPO Annie Chechitelli said Turnitin "researched and identified the signals and patterns of leading humanizers and have trained our model to identify them", and that humanizers "leave a statistical trace that can be learned and detected." Plagiarism Today notes that humanizers typically edit common AI phrases and punctuation such as em dashes. No detection rates for bypassed text were published, and the only figure repeated was the general FPR of under 1% — [Plagiarism Today, Aug 27, 2025](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/)
- Timeline: AI detection Feb/Apr 2023, AI-paraphrase detection July 2024, Turnitin Clarity (process-based writing environment) 2025, bypasser detection Aug 2025 — [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/)

**Independent results on humanized text:**
- In the VUB study (Int. J. Educational Integrity, 2026; 160 documents of known origin), Turnitin, GPTZero and Copyleaks "significantly underestimated GenAI content, particularly for texts generated with the most advanced model". Pangram outperformed them on hybrid and humanised texts — [AI Weekly summary, Oct 2026](https://aiweekly.co/alerts/study-finds-pangram-beats-gptzero-turnitin-at-spotting-ai-text); [Springer article page](https://link.springer.com/article/10.1007/s40979-026-00226-w). A search snippet attributed "Turnitin 0.0%" detection on one fully AI-generated condition to the appendix, but I could not verify this.

### Inferences
- Turnitin's pipeline is AIW gating plus AIR labelling. As a result, humanized text that gets past AIW-2's sentence threshold, or keeps the document under 20%, is invisible: it shows as "*%" or as human. Bypasser detection appears to be folded into the AI-generated scoring (Turnitin calls it part of AI writing detection), not a new visible label, but this is not confirmed.
- The mirrored human/AI data generation in Turnitin's training resembles Pangram's "synthetic mirror" approach.

### Gaps
- No published recall or FPR for the Aug 2025 bypasser model. I could not determine how bypasser hits appear in the report (a separate highlight color/category or merged into AI-generated). Turnitin guide pages were blocked (403).
- No independent peer-reviewed test of the bypasser feature was found.

## GPTZero: model evolution, "AI paraphrased" category, humanizer robustness, ESL bias

### Takeaway
GPTZero started from perplexity and burstiness (2023) and is now a deep classifier with a separate "AI paraphrased" sub-class, currently in beta. GPTZero claims 93.5% recall on its own 12-tool paraphrase benchmark. However, the independent Booth/BFI study (Aug 2025) found GPTZero lost roughly half or more of its detections after StealthGPT humanization. The original RAID paper (2024) had found GPTZero unusually robust to character-level attacks.

### Cited Findings
- **Result categories (VENDOR, support page updated Sept 2025):** Advanced Scan outputs "chance AI Generated" and "chance AI Paraphrased" separately. When AI paraphrasing is detected with medium or high confidence, a "Possible AI Paraphrasing" badge appears with the text "We are highly confident this text was originally AI, but rewritten by AI or a human." This covers bypassers/humanizers and manual rewording. GPTZero claims an FPR under 0.1% "for detecting paraphrased text from certain popular AI tools", calls the feature beta, and says bypasser coverage is "still improving." It recommends its Authorship / Google Docs writing-replay tool for provenance — [GPTZero support](https://support.gptzero.me/articles/5593633457-how-does-gptzero-detect-ai-paraphrasing-and-ai-bypassers)
- **Humanizer approach and benchmark (VENDOR, blog Jan 26, 2026):** GPTZero targets "deeper semantic and structural signals beyond surface form" and runs internal systems that simulate adversarial attacks using open- and closed-source paraphrasers. The benchmark has 1,000 paraphrased examples spread evenly over more than 12 tools and all their settings. Sources are GPT-5, GPT-4o, GPT-4.1, Claude and Gemini, across 8 domains. AI recall: GPTZero (model 3.15b) 93.5%, Originality (lite_102) 57.3%, Pangram (v3) 50.2%. GPTZero also claims more than 90% on every paraphraser and more than 98% on unparaphrased AI. It names the open-source "Temp" paraphraser (EMNLP 2025) as one tool that fooled Pangram. Data is available to researchers on request — [GPTZero blog](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)
- **Independent (Jabarian & Imas, NBER w34223 / BFI WP 2025-116, version Aug 26, 2025):** every AI passage was run through StealthGPT's default rewrite endpoint. GPTZero's FNR was "around 0.50 and above across most genres and LLM models", ranging from 0.2552 (Gemini 2.0 Flash resumes) to 0.7730 (Claude Opus 4 novels). The authors conclude GPTZero "largely loses its capacity" on humanized text. On unhumanized text, GPTZero had FPR ≤ 0.0238 per genre at tuned thresholds and about 0.0071 across fixed thresholds, with FNR of 0.002 to 0.030. The study used the API v2 field `average_generated_prob` — [BFI working paper PDF](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf); [NBER](https://www.nber.org/system/files/working_papers/w34223/w34223.pdf)
- **RAID (Dugan et al., ACL 2024), at 5% FPR:** GPTZero scored 66.5% with no attack, 64.0% under paraphrase, 61.0% under synonym and 66.2% under homoglyph. The authors called GPTZero "unusually robust" to adversarial attacks — [RAID arXiv 2405.07940](https://arxiv.org/html/2405.07940v1)
- **ESL bias:** Liang et al. (2023, *Patterns*) found an average 61% FPR across 7 detectors on TOEFL essays, linked to low perplexity. GPTZero says it retrained on L2 writing and reports a 1.1% FPR (1 of 91) on that TOEFL set (VENDOR, not independently replicated) — [GPTZero "5 steps towards responsible AI detection"](https://gptzero.me/news/5-steps-towards-responsible-ai-detection); summary via [casrai.org (secondary)](https://casrai.org/guides/is-gptzero-accurate)
- **Notre Dame study (Karr, Khvatskii, Hua, Chawla; arXiv 2608.11256, Aug-Oct 2026):** at τ = 0.50, light "refine abstract" AI edits were flagged 38-80% of the time, and unmodified 2023-2025 human abstracts 9-15% (much higher for non-STEM, p<0.001). After Undetectable AI humanization, fewer than 4% of AI rewrites stayed flagged. Elevated scores tracked long-token and Academic-Word-List density — [arXiv 2608.11256](https://arxiv.org/abs/2608.11256). A secondary search summary says the detectors were Pangram 3.2 and GPTZero, but the abstract page does not name them.

### Inferences
- The BFI result may understate GPTZero if `average_generated_prob` excludes the "paraphrased" class mass. This is speculative and unverified, but worth noting when comparing it with GPTZero's own 93.5% figure. The two results also differ in time (Aug 2025 vs Jan 2026 models) and humanizer set.

### Gaps
- GPTZero has not published an architecture paper. The current component makeup (any surviving perplexity features vs. a pure deep classifier) is undisclosed.
- No independent replication of GPTZero's 12-tool paraphrase benchmark.

## Originality.ai: model versions, paraphrase/humanizer detection, published tests

### Takeaway
Originality offers several models: Lite, Turbo (aggressive, catches humanized text, higher FPR) and Academic (released around Sept 2025). Vendor claims are about 99% accuracy. Independently, it was mid-tier but fairly robust to StealthGPT in the BFI study (FNR about 0.02-0.21), robust to paraphrase in RAID (accuracy actually rose), and badly broken by homoglyphs in RAID (85.0% to 9.3%).

### Cited Findings
- Model claims (VENDOR, relayed by review sites, not verified on originality.ai): Lite 99% accuracy / 0.5% FP; Turbo 99%+ / 1.5% FP and about 97% detection of humanized text; Academic 99%+ / under 1% FP, released September 2025 — [Plisio review](https://plisio.net/ai/originality-ai); [Sentisight](https://www.sentisight.ai/why-is-originality-ai-held-up-as-the-gold-standard-ai-content-detector/)
- Originality's own humanizer reviews report StealthGPT output scored "Likely AI, 100% confidence" (VENDOR) — [Originality StealthGPT review](https://originality.ai/hi/blog/stealthgpt-ai-review)
- **RAID (2024) at 5% FPR:** Originality scored 85.0% with no attack, 96.7% under paraphrase (+11.7), 96.5% under synonym, 78.6% under misspelling, 9.3% under homoglyph (−75.7) and 71.4% under article deletion. Originality could not get below about 0.62% FPR. Originality promotes this as placing first in 9 of 11 attack tests — [RAID](https://arxiv.org/html/2405.07940v1); [Originality blog (VENDOR framing)](https://originality.ai/blog/robust-ai-detection-study-raid)
- **BFI (2025), using the "Turbo" endpoint:** genre FPR at tuned thresholds was 0.0000-0.0218. At fixed thresholds Originality had the weakest FNR of the commercial tools on unmodified Claude text (Claude Opus 4 FNR 0.30-0.42). Against StealthGPT, FNR was around 0.05 for long text and up to 0.21 for short (e.g., 0.2073 on Gemini restaurant reviews, 0.1905 on GPT-4.1 restaurant reviews) — [BFI WP](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf)
- GPTZero's vendor benchmark put Originality lite_102 at 57.3% on paraphrased text (VENDOR-competitor) — [GPTZero blog](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)

### Inferences
- Originality's improvement under paraphrase in RAID suggests its training data already contained paraphrased or humanized text, consistent with Turbo being marketed as humanizer-aware. Its homoglyph collapse points to tokenization-level fragility rather than stylistic blind spots.

### Gaps
- No primary Originality technical documentation of architecture or training was retrieved. Model figures come via secondary review sites.

## Pangram Labs: architecture, hard-negative mining, mirrored prompts, humanizer robustness, independent evaluations

### Takeaway
Pangram is a fine-tuned LLM classifier trained on "synthetic mirror" pairs plus iterative hard-negative mining. Pangram 4 (2026) adds a MoE backbone, token-level labels including "AI-Assisted", and a dedicated humanizer head. It is the strongest performer in most independent tests on humanized text: BFI/Booth 2025 and VUB 2026. Counter-evidence comes from GPTZero's vendor benchmark (Pangram v3 at 50.2% on paraphrased text) and the Notre Dame 2026 study.

### Cited Findings
- **Original technical report (arXiv 2402.14873, Emi & Spero, Feb 2024):** introduces "hard negative mining with synthetic mirrors". The model scans large datasets for false positives, adds those hard examples to training and retrains over several rounds, reaching a "near-zero false positive rate." For each human document, an AI "mirror" is generated matching its style, tone and content, so the model learns LLM-specific cues rather than topic differences — [arXiv 2402.14873](https://arxiv.org/pdf/2402.14873); [Pangram "how it works"](https://www.pangram.com/research/how-it-works)
- **Humanizer work (DAMAGE, Masrour, Emi, Spero; arXiv 2501.03437, Jan 6 2025, COLING detection workshop):** audits 19 humanizer/paraphrase tools for effect and meaning preservation. It uses data-centric augmentation to train a robust detector and reports cross-humanizer generalization strong enough to survive an attack in which the authors fine-tuned a model against their own detector — [arXiv 2501.03437](https://arxiv.org/abs/2501.03437)
- **Humanizer results (VENDOR blog, Jan 23 2025):** recall on humanized AI text was 93.66% for Pangram with humanizer training, 73.07% for the Pangram baseline, 34.53% for GPTZero and 29.73% for Binoculars. On raw AI text: Pangram 100%, GPTZero 95.60%, Binoculars 94.40%. The claim covers "high-quality" humanizers. No FPR impact or tool names were reported — [Pangram blog](https://www.pangram.com/blog/humanizers-announcement)
- **Pangram 4 Technical Report (arXiv 2607.27183, ~July 2026; read via hyper.ai mirror, not the primary PDF):**
  - Architecture: an open-weight MoE backbone with LoRA adapters; window-level heads for segment edits, mixed authorship and humanizer detection; a token-level head; "Repeat2" input duplication for full context under causal attention; 512-token windows with stride 256; a calibration model plus a 3-state CRF (Viterbi) plus sentence majority vote.
  - Data: human text from licensed or owned corpora only; AI text via synthetic mirroring, with a filter for verbatim copying; AI-assisted text via the EditLens method; hard negatives from an offline active-learning loop (human text misclassified as AI is mirrored and fed back).
  - The humanizer head is trained with stop-gradient so it does not update the backbone.
  - Outputs: Human / AI-Assisted ("Mixed") / AI-Generated at token level, plus a humanizer flag.
  - Results (VENDOR): on a 520k-sample benchmark, AUROC 0.9916, FPR 0.0041%, FNR 0.3396% (Pangram 3.3.2: 1.9942%). Heavily AI-edited essays: AI-assisted recall rose from 5.54% to 55.01%. Humanizer robustness is described only qualitatively.
  — [hyper.ai mirror of arXiv 2607.27183](https://hyper.ai/en/papers/2607.27183)
- API output labels: "AI-Generated", "AI-Assisted", "Human Written" per window — [Pangram API docs](https://docs.pangram.com/api-reference/ai-detection)
- **Independent (BFI/NBER, Jabarian & Imas, Aug 2025; 1,992 human passages × 4 LLMs: GPT-4.1, Claude Opus 4, Claude Sonnet 4, Gemini 2.0 Flash; 6 genres):**
  - Pangram API v2025-05 had FPR 0.0000-0.0075 by genre and 0.0000 at raw thresholds ≥ 0.5. It was the only detector meeting an FPR ≤ 0.005 cap without losing much recall.
  - On StealthGPT-humanized text, FNR was 0.0000 on novels and resumes and at most 0.05 elsewhere (e.g., 0.0400 Claude Opus 4 Amazon reviews, 0.0367 news).
  - Cost per true positive: $0.0228 Pangram vs $0.0416 Originality vs $0.0575 GPTZero.
  — [BFI WP 2025-116](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf); [Chicago Booth Review](https://www.chicagobooth.edu/review/do-ai-detectors-work-well-enough-trust)
- **Independent (VUB: Van Vlasselaer, Van Droogenbroeck, Spruyt; IJEI 2026):** 160 documents of known origin, with a "humanising" prompt meant to mimic student behaviour (no named tool). Pangram outperformed GPTZero, Copyleaks and Turnitin on fully AI, hybrid and humanised texts. False positives were rare for all tools. A secondary snippet reports that on humanised texts (ground truth 44.8% AI) GPTZero estimated 21.2% and Pangram 43.6%. Applied to 1,163 master's theses (2024-25), Pangram flagged 45.5%, mostly at low to moderate levels — [AI Weekly](https://aiweekly.co/alerts/study-finds-pangram-beats-gptzero-turnitin-at-spotting-ai-text); [Springer](https://link.springer.com/article/10.1007/s40979-026-00226-w)
- **Counter-evidence:**
  - GPTZero's vendor benchmark: Pangram v3 50.2% recall on paraphrased text — [GPTZero blog](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)
  - Notre Dame 2026: under 4% of Undetectable-AI-humanized rewrites stayed flagged at τ = 0.5 (detector identity per secondary summary: Pangram 3.2 and GPTZero) — [arXiv 2608.11256](https://arxiv.org/abs/2608.11256)
  - Humanizer-vendor benchmark (Oct 2026, low reliability): Pangram caught 348 of 420 outputs from 14 humanizers (82.9%) — [search-result summary of humanizerbench/phrasly pages](https://humanizerbench.com/detectors/pangram/)

### Inferences
- Pangram's advantage on humanized text comes mostly from data, not features: humanizer outputs in training, mirrored negatives that remove topic shortcuts, and active learning on hard human negatives to keep FPR near zero. Results then depend heavily on whether a particular humanizer (e.g., the Temp paraphraser or new tools) was in the training distribution.

### Gaps
- I could not access the primary Pangram 4 PDF to confirm the figures or get humanizer-specific numbers.
- The official RAID leaderboard entry for Pangram was not retrieved.

## Copyleaks, Winston AI, ZeroGPT, Sapling, Hive (brief)

### Takeaway
Reliable independent data on these tools is thin. RAID (2024) found Winston and ZeroGPT weak under paraphrase and ZeroGPT unable to reach low FPR. Copyleaks underestimated AI content in the VUB 2026 study. I found nothing credible on humanized text for Sapling or Hive.

### Cited Findings
- **RAID:** ZeroGPT's FPR could not go below 16.9%. Per secondary summaries of RAID's table, under paraphrase attack at 5% FPR Winston scored 52.6% and ZeroGPT 46.7% (Winston 71.0% with no attack) — [RAID arXiv](https://arxiv.org/html/2405.07940v1) (FPR plateau confirmed; per-attack Winston/ZeroGPT numbers came from a search summary, not the fetched excerpt)
- **Winston (VENDOR):** claims 99.89% detection under RAID attacks at 1% FPR and 99.98% on its own 10k-text test — [Winston benchmark page](https://gowinston.ai/most-accurate-ai-detector-across-leading-benchmarks/)
- **Copyleaks:** underestimated AI share in the VUB 2026 study — [AI Weekly](https://aiweekly.co/alerts/study-finds-pangram-beats-gptzero-turnitin-at-spotting-ai-text). No false positives on a small sample in arXiv 2507.17944 (garbled snippet, low confidence) — [arXiv 2507.17944](https://arxiv.org/pdf/2507.17944)
- **ZeroGPT:** Erol et al. (2025) reported a 16% FPR on human text, the highest among the tools tested (via secondary summary) — [fast.io summary](https://fast.io/resources/ai-detector-accuracy-comparison-2026/)

### Gaps
- No primary method disclosures were found for Copyleaks, Winston, ZeroGPT, Sapling or Hive. No independent humanizer tests were found for Sapling or Hive.

## Signals used against humanized text, and what still fools detectors

### Takeaway
Vendors describe learned statistical "traces" left by paraphraser architectures, not hand-written rules. Learned deep classifiers have largely replaced hand-crafted perplexity and burstiness features. Character-level tricks such as homoglyphs historically broke some detectors badly. Strong, meaning-changing humanizers (StealthGPT, Undetectable AI) and newer models still evade some detectors at high rates.

### Cited Findings
- **Turnitin:**
  - Paraphrasers "use a different generative architecture framework from a standard LLM, and therefore leave a distinct and detectable signature".
  - Transformers capture "higher order deviations" beyond perplexity and burstiness.
  - Window statistics need about 5-10 sentences.
  — [Turnitin white paper](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf). Humanizers "leave a statistical trace"; they typically strip stock AI phrases and em dashes — [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/)
- **GPTZero:** "deeper semantic and structural signals beyond surface form". Humanizers are described as altering syntax and vocabulary and accepting grammatical errors — [GPTZero blog](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)
- **Pangram:** mirrors force reliance on LLM-specific stylistic cues, plus a dedicated humanizer head and token-level edit detection — [Pangram 4 report mirror](https://hyper.ai/en/papers/2607.27183); [Pangram how-it-works](https://www.pangram.com/research/how-it-works)
- **Attack sensitivity (RAID 2024):**
  - Homoglyphs cut detectors by an average of 40.6 points (Originality 85.0 → 9.3). GPTZero was robust (−0.3).
  - Paraphrase and synonym attacks raised Originality's accuracy.
  - Article deletion −13.6 and misspelling −6.4 for Originality.
  — [RAID](https://arxiv.org/html/2405.07940v1)
- **What still fools detectors:**
  - StealthGPT defeats GPTZero (FNR about 0.5 or higher) — [BFI](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf)
  - Undetectable AI cut flags to under 4% — [Notre Dame 2026](https://arxiv.org/abs/2608.11256)
  - Short texts (under 50 words) hurt all detectors — [BFI](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf)
  - Scores correlate with academic vocabulary density, so formal human writing and lightly AI-edited text get flagged — [Notre Dame 2026](https://arxiv.org/abs/2608.11256)
  - AI-assisted/edited text remains hard: Pangram 4 recall on heavily AI-edited essays is only 55.01% (VENDOR) — [Pangram 4 mirror](https://hyper.ai/en/papers/2607.27183)

### Inferences
- The industry's shared anti-humanizer recipe is: (1) run commercial humanizers on AI text to make training positives; (2) add mirrored and hard-negative human text to keep FPR low; (3) use window or token-level scoring so partial rewrites still trip local thresholds. Humanizers that win are typically new tools not yet in the training data, tools that rewrite heavily enough to change the token distribution, or targeted attacks against a specific detector.

### Gaps
- No vendor publicly lists specific hand-crafted features (e.g., "unnatural synonym" detectors or character-normalization steps). Whether Turnitin, Pangram or GPTZero normalize Unicode against homoglyphs today is undisclosed.

## Independent head-to-head studies 2023-2026 (numbers)

### Takeaway
The most rigorous recent independent comparisons are BFI 2025 and VUB 2026, and both rank Pangram first, especially on humanized text. RAID (2024) remains the largest adversarial benchmark, but it uses older detector versions. Weber-Wulff et al. (2023) set the "neither accurate nor reliable" baseline. Notre Dame (2026) argues that detectors penalize honest AI editing more than humanizer evasion.

### Cited Findings
- **Weber-Wulff et al. (2023, IJEI; arXiv 2306.15666):** 14 tools including Turnitin and PlagiarismCheck. "Neither accurate nor reliable". Errors skew toward labelling AI text as human, and obfuscation "significantly worsens" performance — [arXiv 2306.15666](https://arxiv.org/abs/2306.15666)
- **RAID (Dugan et al., 2024):** 6,287,820 texts (509,014 non-adversarial generations from 14,971 human docs), 11 attacks, evaluated at a fixed 5% FPR. The authors argue FPR must be fixed and reported for comparability — [RAID](https://arxiv.org/html/2405.07940v1)
- **Jabarian & Imas (NBER w34223 / BFI, Aug 2025):** see the Pangram/GPTZero/Originality sections. RoBERTa-base FPR was 0.31-0.78 by genre. Pangram was the only tool to meet the FPR ≤ 0.005 cap — [BFI WP](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf)
- A GPTZero-run re-analysis (Jan 2026; 8,697 docs) reportedly found GPTZero and Pangram near 99% and Originality at 85.0% (VENDOR re-run, secondary summary only) — [search summary; GPTZero news tag page](https://www.gptzero.me/news/tag/ai-detection/page/2/)
- **VUB (IJEI 2026):** 160 documents; Pangram > Turnitin, Copyleaks, GPTZero; rare FPs overall — [Springer](https://link.springer.com/article/10.1007/s40979-026-00226-w)
- **Notre Dame (arXiv 2608.11256, 2026):** 9-15% flags on recent human abstracts; 38-80% on light AI edits; under 4% after humanization — [arXiv](https://arxiv.org/abs/2608.11256)

### Gaps
- No access to Weber-Wulff per-tool numbers or the full VUB tables (Springer was paywalled or redirected). No current official RAID leaderboard snapshot was retrieved. No independent study was found that tests Turnitin's Aug 2025 bypasser feature specifically.
