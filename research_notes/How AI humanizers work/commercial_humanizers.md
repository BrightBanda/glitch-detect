# Commercial AI Humanizers / AI-Bypass Tools: Methods, Features and Measured Effectiveness (2024-2026)

Research date: 2026-10-26. Evidence-quality labels used below: **[VENDOR]** = claim by the tool maker; **[DETECTOR-VENDOR]** = test published by a company that sells a detector (conflict of interest); **[COMPETITOR]** = test by a rival humanizer seller; **[ACADEMIC]** = paper or working paper; **[JOURNALISM]** = news or trade press; **[WEAK]** = affiliate blog, SEO page or forum.

Warning on sources: most search results for "best AI humanizer" are SEO and affiliate pages. Some are on huggingface.co profile pages (e.g. "humanize-ai-text-2026", "turnitin-humanizer-2026"), and these give numbers that contradict each other for the same tool (HumanizeMyPaper is listed both as "3.8% Turnitin AI" and as a "96% Turnitin pass rate"). Many vendor blogs rank the author's own product first. The notes below exclude those numbers except where they are labelled as examples of marketing.

---

## 1. QuillBot (Paraphraser + AI Humanizer) and how detectors treat paraphrased AI text

### Takeaway
QuillBot has not published how its paraphraser or humanizer models work. They are best understood as LLM or seq2seq paraphrasers that keep sentence structure close to the source. Major detectors now target this kind of output directly: Turnitin has a separate "AI-generated text that was AI-paraphrased" category, and GPTZero has an "AI paraphrased" sub-class. Pangram reports catching 100% of QuillBot-humanized text (Aug 2025, detector-vendor test).

### Cited Findings
- QuillBot AI Humanizer has two tiers. Free "Basic" humanizing is capped at 125 words per input. "Advanced" humanizing is Premium-only, advertised as "deeper rewrites and improved fluency" with no word or usage limit **[VENDOR]** — [QuillBot product page via search snippet](https://quillbot.com/ai-humanizer?pp=env); Scribbr's 2026 guide lists Premium at about $8.33/month billed annually **[WEAK/affiliate-adjacent; Scribbr is QuillBot's sister brand under Learneo]** — [Scribbr](https://scribbr.com/ai-tools/best-ai-humanizer)
- QuillBot also sells an AI detector. It entered AI detection around July 2024, so it sells both evasion and detection — [EmergentMind summary](https://www.emergentmind.com/topics/quillbot) **[WEAK/secondary]**
- In Pangram's DAMAGE paper (Jan 2025), QuillBot output was rated "L1" (best quality tier): "flowery but readable; vocabulary is high but slightly imprecise." The paper groups QuillBot with Grammarly and DIPPER as paraphrasers rather than purpose-built humanizers **[DETECTOR-VENDOR/ACADEMIC, peer-reviewed workshop]** — [Masrour, Emi, Spero, arXiv:2501.03437 / GenAIDetect 2025](https://arxiv.org/html/2501.03437v1)
- Pangram's humanizer benchmark (updated Aug 2025) reports detection accuracy on QuillBot-processed text of 100.0% **[DETECTOR-VENDOR]** — [Pangram blog](https://www.pangram.com/blog/humanizers-aug-25)
- Turnitin's AI writing report splits the AI score into two categories: "AI-generated only" (cyan highlight) and "AI-generated text that was AI-paraphrased" (purple highlight), meaning text likely produced by an LLM and then changed with a paraphrasing tool. Turnitin warns the model "may misidentify human-written, AI-generated, and AI-generated and AI-paraphrased text" and should not be the sole basis for adverse action — [Turnitin guide: How should I review the AI Writing report](https://guides.turnitin.com/hc/en-us/articles/27139000787853-How-should-I-review-the-AI-Writing-report); [Turnitin classic report view guide](https://guides.turnitin.com/hc/en-us/articles/28457596598925-AI-writing-detection-in-the-classic-report-view)
- Turnitin launched its anti-AI-paraphrasing detection in July 2024 — [Plagiarism Today, 27 Aug 2025](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/)
- GPTZero says it flags AI paraphrasers and humanizers, names QuillBot among the tools it targets, and links to an "AI paraphrased sub-class" in its detector **[DETECTOR-VENDOR]** — [GPTZero: AI paraphrasing detection](https://gptzero.me/news/ai-paraphrasing-detection/); [GPTZero, 26 Jan 2026](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)
- A forum user reports that QuillBot's Basic humanizer mode did not lower detector scores in their own tests **[WEAK, anecdotal]** — [forum post](https://forum.network-inventory-advisor.com/t/quillbot-ai-humanizer-review/2280)

### Inferences
- QuillBot is a mainstream "writing assistant" paraphraser rather than a purpose-built evasion tool. Its output stays fluent and close to the source, and Pangram's own data says fluent output is easier to detect (see section 4). Both points fit QuillBot being among the most easily detected tools.
- Turnitin's purple "AI-paraphrased" category is aimed at QuillBot-style paraphrasing. The Aug 2025 "bypasser" update (section 3) extends this to purpose-built humanizers.

### Gaps
- QuillBot's pages returned HTTP 403 to the fetcher, so I could not verify current Paraphraser mode names (historically Standard, Fluency, Formal, Simple, Creative, Expand, Shorten, Academic, Custom, plus a "synonyms" slider) or Humanizer details from a primary source. Treat the mode list as unverified background knowledge.
- I found no technical disclosure (model architecture, training data, or whether QuillBot uses third-party LLMs) for either product.
- I found no independent (non-vendor) numeric test isolating QuillBot Humanizer against Turnitin.

---

## 2. Purpose-built humanizers (Undetectable.ai, StealthGPT, StealthWriter, WriteHuman, HIX Bypass, BypassGPT, Phrasly, Humbot, Grammarly Humanize, etc.): claimed methods and features

### Takeaway
No vendor has published a real technical disclosure: no papers, patents or model cards were found. Claims are marketing phrases such as "proprietary models", "rewrites at the model level addressing perplexity and burstiness", and "tested against major detectors before production". The only systematic look inside comes from Pangram's DAMAGE audit (Jan 2025). It found a split between LLM-based rewriters (system-prompted or fine-tuned), which drift further from the source, and rule-based dictionary/synonym tools. Several popular tools (Undetectable AI, WriteHuman, Humbot, Phrasly, HIX Bypass) produced degraded, sometimes unreadable text.

### Cited Findings

**Taxonomy of internals (inferred by researchers, not disclosed by vendors)**
- DAMAGE audit of 19 humanizer/paraphraser tools (DIPPER, Grammarly, QuillBot, Bypass GPT, Ghost AI, HIX Bypass, Humbot AI, HumanizeAI.io, HumanizeAI.pro, Humanizer.com, Phrasly.ai, Semihuman AI, StealthGPT, StealthWriter.AI, Surfer SEO, Undetectable AI, Twixify, WriteHuman.ai, GPTInf, AIHumanizer.com). The paper's Table 1 (18 names) and Appendix E (19) do not fully match. It found that some humanizers "are LLMs with system prompts or fine-tuning" and are more weakly grounded in the source, while rule-based tools mostly do synonym replacement and stay closer to the original **[DETECTOR-VENDOR/ACADEMIC]** — [arXiv:2501.03437](https://arxiv.org/html/2501.03437v1)
- Pangram (Aug 2025): "Some humanizers add manual spacing and punctuation errors and swap words one-to-one with synonyms. Others are fine-tuned LLMs trained to paraphrase text in a more human style." **[DETECTOR-VENDOR]** — [Pangram](https://www.pangram.com/blog/humanizers-aug-25)
- GPTZero (Jan 2026): humanizers rewrite to reduce "statistical patterns common in AI-generated text", "heavily altering syntax and vocabulary, sometimes adding grammatical inaccuracies" **[DETECTOR-VENDOR]** — [GPTZero](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)
- Turnitin's CPO (Aug 2025): humanizers edit "common AI phrases and punctuation, such as the famous em dash" but "also leave a statistical trace that can be learned and detected" — [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/)

**Per-tool quality ratings, DAMAGE paper Appendix E (Jan 2025; tool versions as of late 2024)** **[DETECTOR-VENDOR/ACADEMIC]** — [arXiv:2501.03437](https://arxiv.org/html/2501.03437v1)
- L1 (best): DIPPER, GPTInf, Grammarly ("occasionally odd edits such as extra quotation marks"), HumanizeAI.pro, QuillBot, Semihuman AI ("adds personal pronouns not in the source"), StealthGPT ("closely matches the original style"), Twixify ("occasional word misuse from dictionary-based replacements").
- L2 (medium): AIHumanizer.com (downgrades university-level text to middle-school level, adds punctuation mistakes), BypassGPT ("dictionary-lookup paraphrasing; words sometimes used in the wrong context"), StealthWriter.AI (about one grammar/punctuation/capitalization error per paragraph), Surfer SEO (middle-school level).
- L3 (worst): Ghost AI (splits every sentence into single-clause statements), HIX Bypass ("occasional dense pockets of nonsensical text"), HumanizeAI.io (invents citations, inserts "??????", punctuation errors), Humbot AI ("uninterpretable sentences... unreadable"), Phrasly (poor sentences, high variance), Undetectable AI ("elementary-level writing with typos"), WriteHuman.ai ("elementary-level, sometimes incomplete sentences").
- Fluency win rate vs the original AI text (GPT-4o as judge): L1 26.0%, L2 14.67%, L3 2.67%. Most humanizers therefore made the text less fluent than the AI original.

**Undetectable.ai**
- Homepage claims: rewrites flagged sections by "changing sentence structure, expanding vocabulary, and adjusting tone"; output scores "as human on every major detector" (detectors not named); 50+ languages ("English has the highest accuracy"); free tier plus paid plans; "our guarantee" (terms not stated on homepage); no academic-use disclaimer on the homepage. No model is named and there is no statement about training against detectors **[VENDOR]** — [undetectable.ai](https://undetectable.ai/) (fetched Oct 2026)
- Third-party listings describe pre- and post-processing, readability formulas (Flesch-Kincaid, Gunning Fog, SMOG), and modes such as "Instant/Advanced/Precise". Pricing figures conflict: about $9.99/10k words, $19/20k, $31/35k per month; or $10 Basic / $20 Pro (50k); trial of 150-250 words **[WEAK, conflicting]** — [fast.io review](https://www.fast.io/resources/undetectable-ai-review-2026.md); [SF Examiner marketplace](https://www.sfexaminer.com/marketplace/undetectable-ai-review-does-it-really-outsmart-ai-detectors/article_78c79b10-a10f-11ee-9117-bba78068a9d6.html); [GetApp](https://www.getapp.com/all-software/a/undetectable-ai/); [creati.ai](https://creati.ai/ai-tools/undetectableai/)
- fast.io reports that readability decreased in 45% of humanized samples in its comparison **[WEAK]** — [fast.io](https://www.fast.io/resources/undetectable-ai-review-2026.md)
- Undetectable.ai also sells AI-text, image, voice and video detectors and "fake receipt/ID" detectors, so it markets itself as both evader and detector **[VENDOR]** — [undetectable.ai](https://undetectable.ai/)

**StealthGPT**
- Claims (fetched Oct 2026) **[VENDOR]** — [stealthgpt.ai](https://www.stealthgpt.ai/):
  - "Completely private and proprietary" models "trained to emulate expert human writing"; "rewrites at the sentence level", and "at the model level, addressing perplexity, burstiness, and every other signal AI detectors measure"; models "tested heavily against major AI detectors before being moved into production".
  - Claims to remove Claude/Gemini watermarks and offers SynthID "removers".
  - Modes: "Stealth Lite" (removes em dashes and stock phrases, stays close to the original) and "Super" (full rewrite). Tones: Standard, High School, College, Ph.D.
  - Built-in "human score" detector and "Enhanced AI Detection" that estimates Turnitin's result. Self-reported "Turnitin: 98% human average (0/30 detections). GPTZero: 95% (0/30)".
  - Demo box limited to 100 words; up to 3,000 words per run (20,000 on Enterprise, chunked in parallel).
  - $1 for 3 days promotion; 99 cents first month with a student email; money-back guarantee "if content gets detected".
  - Markets to students explicitly (homework/quiz answers via Chrome extension, student pricing).
  - Disclaimer: "No tool guarantees 100% bypass, detector models update regularly."
  - Page inconsistencies: "5,000 total languages" vs "250+ languages".
- Other reported pricing: from about $14.99-24.99/month; one-use-per-week free plan **[WEAK, conflicting]** — [bestreview.it.com](https://bestreview.it.com/review/stealthgpt-review/)

**Grammarly "Humanize" + Authorship**
- Grammarly (owned by Superhuman) ships a Humanize feature that "attempts to rewrite AI-generated text to sound more human." In a Nov 2025 test, a 500-word Gemini essay run through Humanize was shown in Grammarly Authorship as "75% Typed by a Human", with fine print showing 0% typed and 75% "Rephrased with Grammarly's AI". Plagiarism Today called this "laundering". On 12 Nov 2025 Superhuman said it would change the display **[JOURNALISM]** — [Plagiarism Today, 6 Nov 2025](https://www.plagiarismtoday.com/2025/11/06/how-grammarly-launders-ai-generated-content/)
- Fixed by Dec 2025: Authorship reports now group content as "Typed by a Human", "Copied from a Source" and "Unknown", with humanized text no longer counted as human-typed — [Plagiarism Today, 11 Dec 2025](https://www.plagiarismtoday.com/2025/12/11/grammarly-updates-authorship-improves-labeling/)
- Grammarly Authorship was announced 14 Aug 2024 (beta in Google Docs from Sept 2024) to show human-typed vs AI-generated vs AI-edited text — [Grammarly](https://www.grammarly.com/blog/company/grammarly-launches-grammarly-authorship/)
- Pangram detection of Grammarly-humanized text: 100.0% (Aug 2025) **[DETECTOR-VENDOR]** — [Pangram](https://www.pangram.com/blog/humanizers-aug-25)

**HIX Bypass, BypassGPT, WriteHuman, Phrasly, Humbot, others**
- BypassGPT is described as "aggressive rewriting" with a "higher risk of meaning drift". Reported pricing is about $10/month annual or $15/month for 50k words **[WEAK]** — [mailmodo](https://www.mailmodo.com/guides/ai-humanizer-tool/); [cocofax](https://cocofax.com/blog/top-10-ai-humanizers-best-tools-to-humanize-ai-text/)
- HIX Bypass sits inside the HIX.AI suite and is described as "inconsistent" **[WEAK]** — [bestreview.it.com](https://bestreview.it.com/review/hix-bypass-review/)
- WriteHuman is described as having an "adaptive rewrite engine" and a built-in "undetectable" score **[WEAK]** — [flowhunt](https://www.flowhunt.io/blog/ai-text-humanizer-comparison/)
- Phrasly (itself a humanizer seller) publishes benchmarks against Pangram claiming rivals fail: StealthGPT passed 38/72 in its test vs StealthGPT's self-reported 89% on "Pangram V4"; WriteHuman averaged 13.3% "human" on Pangram **[COMPETITOR]** — [Phrasly: what Pangram detects](https://phrasly.ai/blog/what-pangram-detects-humanizers); [Phrasly benchmarks, Sept 2026](https://phrasly.ai/benchmarks)

### Inferences
- The vendors' "trained against detectors" language suggests an adversarial loop: generate rewrites, score them with detectors (or their APIs), and keep or fine-tune on the ones that pass. Pangram's DAMAGE paper ran exactly this experiment (fine-tuning GPT-4o on detector-evading pairs, section 3). That suggests the approach is feasible, but no vendor has confirmed it.
- Built-in detector scores (StealthGPT "human score", Undetectable's multi-detector check) are partly marketing. Detector vendors update their models, and humanizer self-checks may lag or use older or free detector tiers.
- Labels such as "readability levels", "High School/College/PhD" tones and "purpose" are most likely prompt or style controls on an LLM rewriter.

### Gaps
- No patents, model cards or engineering blog posts from any humanizer vendor were found. Internals beyond the DAMAGE taxonomy are unknown.
- I did not collect primary-source feature and pricing details for WriteHuman, HIX Bypass, BypassGPT, Phrasly, Humbot, Netus, Walter Writes or Ryne. Only weak affiliate descriptions were found, with conflicting prices.
- Terms of service (e.g. clauses disclaiming academic misuse) were not reviewed for most vendors.

---

## 3. Independent and semi-independent tests against named detectors

### Takeaway
Results depend heavily on the detector. By 2025-2026, Pangram (and, per its own data, GPTZero after updates) catches most humanized text in vendor-run and some academic tests. Other detectors, especially older GPTZero versions, Turnitin on some text types, Copyleaks and Binoculars-style zero-shot methods, miss a large share. However, a 2026 Notre Dame study found that Undetectable AI defeated commercial detectors on research abstracts more than 96% of the time. Results conflict across text genre, length, humanizer version and detector version.

### Cited Findings
- **Jabarian & Imas, "Artificial Writing and Automated Detection", NBER WP 34223 (Sept 2025), not peer-reviewed.** 1,992 passages (news, Amazon reviews, novels, restaurant reviews, summaries; 4 LLMs). Pangram was the only detector to meet a strict FPR ≤ 0.005 policy cap while still detecting AI text, and stayed robust to humanizers. GPTZero's false-negative rate on humanized text was about 50%. Originality.ai separated AI and human text better than GPTZero but had a higher FPR. RoBERTa's FPR was 30-78% — [NBER](https://www.nber.org/papers/w34223); [Chicago Booth Review, Dec 2025](https://www.chicagobooth.edu/review/2025/december/do-ai-detectors-work-well-enough-trust); [Tech & Learning](https://techlearning.com/news/some-ai-detection-tools-work-well-others-fail-says-new-research). Pangram summarizes the paper as: Originality miss rate about 0.05 on long text, up to 0.21 on short; GPTZero missed "around half or more" **[as summarized by DETECTOR-VENDOR]** — [Pangram](https://www.pangram.com/blog/humanizers-aug-25). A secondary source says StealthGPT was the humanizer used, with Pangram missing 0-5% — [humanizerbench/secondary](https://humanizerbench.com/detectors/pangram/) **[WEAK; I could not open the PDF to verify which humanizer was used]**
- **Karr, Khvatskii, Hua & Chawla (Univ. of Notre Dame), "Why AI Detection Fails for Academic Integrity", arXiv:2608.11256 (Aug 2026, rev. Oct 2026; accepted to ACM AI Leadership Summit).** Abstracts from four domains, 2013-15 vs 2023-25. After Undetectable AI humanization, fewer than 4% of AI-labeled rewrites stayed flagged at a threshold of 0.50 (FNR > 96%). Light "refine abstract only" edits were flagged 38-80%, so honest AI editing carried a higher sanction risk than humanizer evasion. A secondary summary says the detectors were Pangram 3.2 and GPTZero and the humanizer was Undetectable AI v11; the abstract page does not name them — [arXiv:2608.11256](https://arxiv.org/abs/2608.11256); secondary detail via [mexc news summary](https://www.mexc.com/news/118832) **[WEAK for the version details]**
- **Van Vlasselaer, Van Droogenbroeck & Spruyt (VUB), Int. J. for Educational Integrity (2026), peer-reviewed.** 160 documents of known origin (human, AI, hybrid, and "humanised" AI rewritten with a student-like prompt, not a commercial humanizer). On humanised texts (ground truth 44.8% AI), Pangram scored 43.6% and GPTZero 21.2% (lowest). On fully AI texts the appendix reports Turnitin 0.0%, GPTZero 14.3%, Copyleaks 14.0% and Pangram 85.7%. On human texts, Turnitin, Copyleaks and Pangram were 0% and GPTZero had a 1.9% false positive — [Springer, s40979-026-00226-w](https://link.springer.com/article/10.1007/s40979-026-00226-w); [AI Weekly summary](https://aiweekly.co/alerts/study-finds-pangram-beats-gptzero-turnitin-at-spotting-ai-text)
- **DAMAGE (Pangram authors, Jan 2025), 19 humanizers, academic text.** TPR at 5% FPR on humanized AI text: GPTZero 60.04% (vs 99.73% on raw AI), Binoculars 28.23% (vs 94.15%), RADAR 5.05%, DAMAGE 98.26%. At default thresholds: GPTZero humanized TPR 34.53% (FPR 1.47%), Binoculars 29.73%, DAMAGE 97.47% (FPR 3.40%). Adversarial test: GPT-4o fine-tuned on outputs that evaded DAMAGE dropped DAMAGE's default-threshold TPR to 93.2% (100% at 5% FPR), with fluency comparable to an L2 humanizer **[DETECTOR-VENDOR/ACADEMIC]** — [arXiv:2501.03437](https://arxiv.org/html/2501.03437v1)
- **Pangram humanizer table (Aug 2025)** **[DETECTOR-VENDOR]**: Ahrefs, aihumanizer.com, Ghost AI, Grammarly, humanizeai.pro, QuillBot, Semihuman, Smodin, Surfer SEO and surgegraph 100%; BypassGPT 99.7%; GPTinf 99.2%; Scribbr 99.0%; Writesonic 98.1%; DIPPER 97.6%; StealthGPT 95.6%; humanizeai.io 93.8%; JustDone 93.5%; TwainGPT 92.7%; Undetectable AI 90.3%. Sample size and model version are not stated. Also cites Russell et al.: Pangram 97% on humanized text vs GPTZero 46%, FastDetectGPT 23%, Binoculars 7% — [Pangram](https://www.pangram.com/blog/humanizers-aug-25)
- **GPTZero benchmark (26 Jan 2026)** **[DETECTOR-VENDOR]**: 1,000 paraphrased examples from GPT-5, GPT-4o, GPT-4.1, Claude and Gemini, run through 12+ paraphrasers including the open-source "Temp paraphraser" (EMNLP 2025). AI recall: GPTZero 3.15b 93.5%, Originality lite_102 57.3%, Pangram v3 50.2%. This contradicts Pangram's own figures, as each vendor wins its own benchmark — [GPTZero](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/)
- **Turnitin "AI bypasser detection" (launched 27 Aug 2025)**: trained on "signals and patterns of leading humanizers"; folded into the existing AI writing report with no new settings or visuals; English only; reports made before 27 Aug 2025 must be resubmitted. No independent accuracy data at launch — [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/); [Campus Technology, 2 Sept 2025](https://campustechnology.com/Articles/2025/09/02/New-Turnitin-Detection-Feature-Helps-Identify-Use-of-AI-Humanizer-Tools.aspx); [Univ. of Bristol blog, 4 Sept 2025](https://blackboard.blogs.bristol.ac.uk/2025/09/04/turnitin-ai-detection-update)
- Single-sample reviews: one hands-on review found Undetectable AI output scored 100% AI on Pangram; another found WriteHuman scored 100% human on Pangram (July 2026) **[WEAK, n=1, affiliate]** — [fast.io](https://www.fast.io/resources/undetectable-ai-review-2026.md); [detectiondrama](https://detectiondrama.com/does-writehuman-bypass-pangram/)
- A 2026 review roundup reports StealthGPT at 86% human on GPTZero and 93% on ZeroGPT, but 36% AI on Turnitin and 47% AI on Originality.ai **[WEAK]** — [fast.io best humanizer](https://fast.io/resources/best-ai-humanizer/)

### Inferences
- Detector-vendor benchmarks are not mutually consistent. Each vendor (Pangram, GPTZero) reports itself best and the other weaker, so readers should weight the NBER, IJEI and Notre Dame studies more heavily.
- The Notre Dame result (> 96% evasion with Undetectable AI on abstracts, Aug 2026) contrasts with Pangram's 90.3% catch rate on Undetectable AI (Aug 2025). Likely causes are newer humanizer versions, a different genre (short, technical abstracts) and older detector versions. Short text is a known weak point (NBER).
- Evasion effectiveness is a moving target. Detectors retrain on humanizer output (Turnitin Aug 2025, GPTZero 2026), and humanizers retrain or re-prompt in response.

### Gaps
- I found no rigorous independent 2025-2026 test that runs many named commercial humanizers against Turnitin specifically. Turnitin access is institutional, so most "Turnitin pass rate" claims come from humanizer vendors or affiliates and are unverifiable.
- I found no major outlet test (Wired, The Verge, Washington Post, Inside Higher Ed) in this search pass. The Forbes piece (3 Oct 2025, "Students Use 'AI Humanizer' Apps To Make ChatGPT Essays Undetectable") returned 403. A snippet says it notes Vanderbilt disabled Turnitin's AI detector over bias concerns — [Forbes](https://www.forbes.com/sites/larsdaniel/2025/10/03/students-use-ai-humanizer-apps-to-make-chatgpt-essays-undetectable/)
- I found no independent quantitative meaning-preservation metric (e.g., BERTScore or human semantic-equivalence ratings) across commercial humanizers. DAMAGE's GPT-4o fluency win rate and qualitative tiers are the best available.

---

## 4. Common tricks and whether detectors catch them

### Takeaway
Documented tricks include synonym spinning (producing "tortured phrases"), deliberate typos and punctuation or spacing errors, sentence splitting, downgrading the reading level, removing em dashes and stock phrases, inserting odd Unicode characters (e.g. thin spaces) to disrupt tokenizers, invented citations and filler, and LLM rewriting tuned for "burstiness" and perplexity. Detector vendors say they now train on these artifacts. Pangram reports a counter-intuitive pattern: the less fluent the output, the less likely it is to be detected, so evasion and quality trade off directly.

### Cited Findings
- Synonym swaps produce "tortured phrases" (e.g. "counterfeit consciousness" for "artificial intelligence"). Repeated tortured phrases suggest systematic substitution. Pangram notes non-native writing can look similar **[DETECTOR-VENDOR]** — [Pangram](https://www.pangram.com/blog/humanizers-aug-25)
- Spacing and Unicode tricks: humanizers remove spaces between sentences, insert unusual spacing, or use characters such as the thin space (U+2009) "to confuse detectors' tokenizers". Deliberate spacing and punctuation errors are also used **[DETECTOR-VENDOR]** — [Pangram](https://www.pangram.com/blog/humanizers-aug-25)
- "More fluent humanizer output is more likely to be detected"; some output is "so garbled that it barely reads as English", which Pangram treats as a separate category it does not try to detect **[DETECTOR-VENDOR]** — [Pangram](https://www.pangram.com/blog/humanizers-aug-25)
- Sentence splitting into single-clause statements (Ghost AI), reading level downgraded to middle or elementary school (AIHumanizer.com, Surfer SEO, Undetectable AI, WriteHuman), fabricated citations and "??????" placeholders (HumanizeAI.io), and added personal pronouns (Semihuman) — [DAMAGE, arXiv:2501.03437](https://arxiv.org/html/2501.03437v1)
- Removing em dashes and stock AI phrases (StealthGPT "Stealth Lite"; Turnitin CPO on humanizers generally) — [StealthGPT](https://www.stealthgpt.ai/); [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/)
- Explicit perplexity and burstiness targeting is claimed by StealthGPT **[VENDOR]** — [StealthGPT](https://www.stealthgpt.ai/)
- Watermark removal: StealthGPT claims to strip Claude and Gemini (SynthID) watermarks **[VENDOR, unverified]** — [StealthGPT](https://www.stealthgpt.ai/)
- Detector responses: Turnitin trained on humanizer "statistical traces" (Aug 2025) — [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/). GPTZero simulates adversarial attacks and trains on "deeper semantic and structural signals" — [GPTZero](https://gptzero.me/news/detecting-ai-humanized-text-how-gptzero-stays-ahead/). Pangram/DAMAGE uses data augmentation with humanizer outputs and generalizes across humanizers, including a detector-tuned GPT-4o — [arXiv:2501.03437](https://arxiv.org/html/2501.03437v1)

### Inferences
- Character-level tricks (homoglyphs, zero-width or thin spaces) are cheap to neutralize with Unicode normalization before scoring. Their presence is itself an evasion signal, so they likely work mainly against naive or zero-shot detectors.
- Deliberate errors and a lower reading level evade detectors partly because they move text toward the distribution of weaker human writers. This raises false-positive and false-negative risks for non-native writers.

### Gaps
- No source found quantifies homoglyph or zero-width-character prevalence in current commercial humanizers, or measures specific detectors' robustness to them.
- No primary documentation found on whether Turnitin, Originality.ai or Copyleaks normalize Unicode before scoring.

---

## 5. Business positioning, ethics and institutional pushback

### Takeaway
Humanizers market to students openly (student-email discounts, homework/quiz extensions, "Turnitin" in marketing copy) while adding "no guarantee" disclaimers. Several sell detectors alongside humanizers. The mainstream writing-tool vendors (Grammarly, QuillBot) frame their tools as making text "natural" or "readable", but Grammarly faced criticism for labelling humanized AI text as human. Institutions have responded with detector features aimed at bypassers, process-based authorship tools, and in some cases by switching AI detection off.

### Cited Findings
- StealthGPT: student-email offer (99 cents first month), Chrome extension for "quiz and homework answers", money-back guarantee "if content gets detected", and a disclaimer that "No tool guarantees 100% bypass" **[VENDOR]** — [StealthGPT](https://www.stealthgpt.ai/)
- Undetectable.ai's homepage has no academic-use disclaimer. It promotes a "guarantee" and sells a suite of detectors (AI text, image, voice, deepfake, fake receipts/IDs) **[VENDOR]** — [undetectable.ai](https://undetectable.ai/)
- Undetectable's flag-refund guarantee depends on which detector and threshold counts as "flagged" — [fast.io](https://www.fast.io/resources/undetectable-ai-review-2026.md) **[WEAK]**
- Grammarly Humanize output was displayed as "Typed by a Human" in Authorship. Plagiarism Today called this "laundering" (Nov 2025), and Grammarly changed its labelling by Dec 2025 — [Plagiarism Today Nov 2025](https://www.plagiarismtoday.com/2025/11/06/how-grammarly-launders-ai-generated-content/); [Dec 2025](https://www.plagiarismtoday.com/2025/12/11/grammarly-updates-authorship-improves-labeling/)
- Turnitin's countermeasures: AI writing detection (Feb 2023), anti-AI-paraphrasing (July 2024), Turnitin Clarity controlled writing environment (May 2025), AI bypasser detection (Aug 2025). Plagiarism Today describes the bypasser feature as the first major plagiarism service to target humanizers specifically — [Plagiarism Today](https://www.plagiarismtoday.com/2025/08/27/turnitin-launches-anti-ai-humanizer-feature/)
- Vanderbilt disabled Turnitin's AI detector over disproportionate flagging of non-native speakers (as noted in Forbes, Oct 2025, snippet only) — [Forbes](https://www.forbes.com/sites/larsdaniel/2025/10/03/students-use-ai-humanizer-apps-to-make-chatgpt-essays-undetectable/)
- The Notre Dame paper concludes that "detector scores should not serve as standalone misconduct evidence" and that honest AI editing faces higher sanction risk than humanizer evasion — [arXiv:2608.11256](https://arxiv.org/abs/2608.11256). The VUB/IJEI paper likewise warns against detectors as sole evidence; applying Pangram to 1,163 master's theses (2024-25) flagged 45.5%, mostly at low to moderate levels — [AI Weekly summary](https://aiweekly.co/alerts/study-finds-pangram-beats-gptzero-turnitin-at-spotting-ai-text) **[secondary]**

### Inferences
- The market has an arms-race structure: humanizers sell "bypass" and self-check detectors, detector vendors publish humanizer benchmarks as marketing, and both sides cite conflicting numbers. Independent academic work (NBER, VUB, Notre Dame) is the best neutral evidence and shows highly detector- and genre-dependent outcomes.
- The ethical risk is asymmetric. Humanizers lower detection of deliberate cheating while honest light AI editing still gets flagged, which strengthens the case for process-based evidence (Authorship, Clarity, version history) over score-based judgments.

### Gaps
- I did not review full terms of service for each vendor (e.g., clauses forbidding academic dishonesty or disclaiming liability).
- I found no reporting on legal or regulatory action against humanizer vendors, or on university bans of specific humanizer sites.
- No revenue or user-count figures were found for humanizer vendors.
