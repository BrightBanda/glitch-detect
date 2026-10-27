# Academic Literature on Evading AI-Generated-Text Detectors (Humanization, Paraphrase Attacks, Adversarial Rewriting)

Scope note: these are a researcher's notes for a report writer. Numbers come from primary papers, fetched from arXiv HTML/PDF and abstracts. Where only an abstract or a secondary summary was available, that is flagged. Timeline for "old vs modern" detectors: OpenAI's classifier (retired July 2023), DetectGPT (2023), and GPTZero/Originality versions tested in 2023-2024 should be treated as historical. Binoculars, Fast-DetectGPT, RADAR, MAGE, Pangram, and the 2025 commercial tools are the more modern reference points.

## 1. DIPPER (Krishna et al., NeurIPS 2023): architecture, results, retrieval defense

### Takeaway
DIPPER is an 11B T5-XXL paraphraser that works on whole paragraphs, can use context, and has lexical (L) and order (O) diversity knobs. One pass at L60/O60 cut DetectGPT from 70.3% to 4.6% and the watermark from 100% to 57.2% (TPR at 1% FPR) while keeping P-SP similarity around 0.94. A retrieval defense on the provider side, where the provider stores its own generations and matches with BM25, still caught about 95-98% of the paraphrases.

### Cited Findings
- **Architecture.** DIPPER fine-tunes a seq2seq Transformer initialized from the pretrained 11B T5-XXL checkpoint on 6.3M paraphrase pairs. — [Krishna et al. 2023, arXiv 2303.13408](https://arxiv.org/abs/2303.13408)
- **Training data.** The pairs come from aligned sentences in different English translations of the same non-English novel (PAR3 dataset).
  - Sentences are aligned with Needleman-Wunsch alignment plus a semantic-similarity metric.
  - Target sentence subsets are randomly shuffled so the model learns content reordering.
  - Source: [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408)
- **Control codes.**
  - Lexical diversity L is computed from unigram token overlap (F1).
  - Order diversity O is computed from the Kendall-Tau correlation of overlapping tokens.
  - Both are bucketed to {0, 20, 40, 60, 80, 100}; "L = 20 roughly corresponds to a 20% lexical modification."
  - Input format: `lexical = L, order = O ⊕ left context ⊕ <p> text </p> ⊕ right context`.
  - Source: [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408)
- **How the attack was run.**
  - Long sequences were paraphrased three sentences at a time, keeping already-paraphrased text in context.
  - Each generation was paraphrased only once (no resampling until evasion), and outputs were 300 tokens.
  - Thresholds were set at 1% FPR on human text.
  - Source: [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408)
- **Table 1: open-ended Wikipedia continuation, detection accuracy at 1% FPR.**
  - **GPT2-1.5B, unattacked:**
    - Watermark 100.0, DetectGPT 70.3, OpenAI classifier 21.6, GPTZero 13.9, RankGen 13.5.
  - **GPT2-1.5B + DIPPER L60/O60 (Sim 94.3):**
    - Watermark 57.2, DetectGPT 4.6, OpenAI 14.8, GPTZero 1.2, RankGen 28.5.
  - **OPT-13B, unattacked:**
    - Watermark 99.9, DetectGPT 14.3, OpenAI 11.3, GPTZero 8.7.
  - **OPT-13B + L60/O60 (Sim 94.6):**
    - Watermark 52.8, DetectGPT 0.3, OpenAI 10.0, GPTZero 1.0.
  - **GPT-3.5 davinci-003, unattacked:**
    - DetectGPT 26.5 (measured at 20% FPR, because it scored 0% at 1% FPR), OpenAI 30.0, GPTZero 7.1.
  - **GPT-3.5 + L60/O60 (Sim 88.4):**
    - DetectGPT 4.5, OpenAI 15.6, GPTZero 1.8.
  - Source: [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408)
- **Baseline strength matters.** Even before any attack, GPTZero (13.9 / 8.7 / 7.1) and the OpenAI classifier were weak at 1% FPR in early 2023. The attack mostly mattered for the watermark and DetectGPT. — [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408)
- **Long-form QA results.**
  - GPT2-XL: watermark 100 → 55.8, DetectGPT 74.9 → 7.6, OpenAI 59.2 → 32.7, at L60/O60 with Sim 96.2.
  - davinci-003: DetectGPT 67.0 → 14.0 (at 20% FPR).
  - Source: [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408)
- **Semantic preservation.**
  - P-SP scores 0.09 on random topically similar paragraph pairs and 0.76 on average for real human paraphrase pairs in PAR3.
  - 88-99% of DIPPER paraphrases score above the median for human paraphrases.
  - Source: [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408)
- **Retrieval defense (long-form QA, 1% FPR).**
  - Mechanism: the API provider stores every generation and scores a candidate by its similarity to the best retrieved match.
  - Detection at L60/O60 with BM25 over the model's own 3K generations: GPT2-XL 97.8, OPT-13B 97.3, GPT-3.5 96.2. The best competing detector reached 55.8 / 51.4 / 38.1.
  - BM25 beat the P-SP retriever: 95.2 vs 75.4 on the pooled 9K corpus for GPT2-XL.
  - At scale (15M T5-XXL generations), it detected 80-97% of paraphrases at 1% human FPR.
  - Source: [Krishna et al. 2023 PDF](https://arxiv.org/pdf/2303.13408); [abstract](https://arxiv.org/abs/2303.13408)
- **The OpenAI classifier no longer exists.** It "was taken down in July 2023 by OpenAI due to its low accuracy." — [Krishna et al. 2023 PDF footnote](https://arxiv.org/pdf/2303.13408)
- **DIPPER against a production watermark.** DIPPER reduced SynthID watermark detection at 1% FPR from 66.5% to 1.5%, as reported in the 2025 DAMAGE paper. — [Masrour et al. 2025](https://arxiv.org/html/2501.03437v1)

### Inferences
- DIPPER's numbers against GPTZero and the OpenAI classifier are from early 2023 and are mainly historical. In RAID (2024), DIPPER-style T5-11B paraphrasing *raised* accuracy for Originality (+11.7) and RoBERTa-GPT2 (+16.2), probably because commercial detectors began training on paraphrased data. See section 4.
- Retrieval only works for the provider whose outputs are stored. It does nothing against open-weight models or third-party detectors, and in Sadasivan et al. it broke down after repeated (recursive) paraphrasing; see section 2.

### Gaps
- I did not fetch DIPPER's human evaluation of paraphrase quality (Appendix C).

## 2. Recursive paraphrasing (Sadasivan et al., "Can AI-Generated Text be Reliably Detected?", 2023; TMLR 2025 version)

### Takeaway
Paraphrasing the same text repeatedly (pp1 to pp5) with DIPPER or LLaMA-2 drives watermark TPR at 1% FPR from 99.8% to below 20% and breaks retrieval (below about 60%). The quality cost is modest: perplexity rises from 5.2 to about 8-10, and 70-83% of outputs are rated high on content preservation. The paper also proves AUROC ≤ 1/2 + TV − TV²/2, so detection becomes impossible in principle as the AI and human text distributions converge.

### Cited Findings
- **Method.** The output of paraphraser P is fed back into P, up to five rounds (pp1-pp5). Paraphrasers tested: DIPPER, LLaMA-2-7B-Chat, and T5. Passages are about 300 tokens. — [Sadasivan et al., arXiv 2303.11156v4](https://arxiv.org/html/2303.11156v4)
- **Soft watermark (Kirchenbauer et al.), TPR at 1% FPR.**
  - Unattacked: 99.8%.
  - After pp1: 80.7% (DIPPER), 54.6% (LLaMA-2).
  - After pp2: below 50%. After pp5: below 20%.
  - Best-of-pp5 selection scored with OPT-1.3B: 4.0%, at a perplexity cost of about 1.5.
  - Source: [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)
- **DetectGPT.** AUROC 96.5% → 59.8% after a T5 paraphrase. — [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)
- **OpenAI RoBERTa-Large detector.** TPR at 1% FPR starts at 100%. The paper reports this inconsistently:
  - Section 2.4 says about 92% after one paraphrase and 80% with ten queries.
  - The introduction says 60%.
  - Source: [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)
- **Retrieval defense (Krishna et al.).**
  - Stays near 100% after pp1 and falls below about 60% after pp5.
  - Perplexity rises by only about 1.7.
  - Source: [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)
- **Quality cost.**
  - Perplexity (OPT-13B): AI text 5.2. DIPPER pp1-pp5: 7.7, 7.8, 8.5, 7.7, 8.7. LLaMA-2 pp1-pp5: 8.1, 9.3, 9.0, 10.3, 10.5.
  - QA accuracy on the content stays at 95.5-97%.
  - Source: [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)
- **Human evaluation (MTurk, share rated 4-5 out of 5).**
  - DIPPER: content preservation 70.4% (the introduction says 77%, an inconsistency); grammar/quality 89.1%.
  - LLaMA-2: content 83.0%; grammar 88.0%.
  - Source: [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)
- **Theorem 1.** AUROC(D) ≤ 1/2 + TV(M,H) − TV(M,H)²/2, where TV is the total variation distance between the machine and human text distributions. As TV → 0, the best possible detector approaches random (AUROC → 0.5). — [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)
- **Spoofing.**
  - The attacker estimated watermark green lists for 181 common words using about 1M queries.
  - With those lists, human-composed text could be made to look watermarked; the paper reports detection AUROC falling from 99.8% to 1.3% in this setting.
  - The retrieval detector could be spoofed 100% of the time.
  - Source: [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)

### Inferences
- The TV bound is the theoretical core of the "humanization always wins eventually" argument. It is a per-document bound: it says nothing about detection across multiple samples from the same source. Rivera Soto et al. (2025, section 3) show multi-sample detection remains feasible.
- The watermark and retrieval results still apply to current watermarks such as SynthID-style schemes, because they attack the mechanism itself. The results against neural classifiers are against 2023 detectors.

### Gaps
- I did not fetch the counter-argument paper by Chakraborty et al. (2023, "On the Possibilities of AI-Generated Text Detection"), which argues that sample complexity makes detection possible with more tokens or samples.

## 3. Adversarially / RL-trained humanizers and optimization attacks (2023-2026)

### Takeaway
The strongest recent attacks either optimize a paraphraser with RL or DPO against detector scores (AuthorMist, StealthRL, DPO-against-Fast-DetectGPT) or steer decoding token by token with a detector (Adversarial Paraphrasing, NeurIPS 2025). They push TPR at 1% FPR to near zero on open detectors, including held-out ones such as Binoculars. They pay a quality or semantic cost:
- StealthRL: LLM-judge quality 2.59/5 vs 4.01 for plain paraphrase.
- Adversarial Paraphrasing: SBERT similarity 0.81 vs 0.86 for plain paraphrase.
- AuthorMist: GPT-2 perplexity up to 300-400.

Style-based detectors and detectors trained on humanizer outputs (Pangram/DAMAGE) are the main exceptions that held up.

### Cited Findings

**RADAR (Hu, Chen, Ho; NeurIPS 2023): adversarial training on the defense side**
- A paraphraser and a detector are trained jointly: the paraphraser tries to evade, and the detector is updated on its outputs.
- It was evaluated on text from 8 LLMs (Pythia, Dolly, Palmyra, Camel, GPT-J, LLaMA, Vicuna) and transferred to GPT-3.5-Turbo. Its abstract says it "significantly outperforms" baselines under paraphrasing.
- Source: [Hu et al., arXiv 2307.03838](https://arxiv.org/abs/2307.03838)
- In RAID, RADAR was among the most attack-stable detectors (drops of 1.4-11.6 points). The RAID authors attribute this to adversarial training. — [Dugan et al. 2024](https://arxiv.org/html/2405.07940v2)
- Its absolute accuracy is low in 2025 evaluations:
  - TPR 5.05% at 5% FPR on humanized academic text (DAMAGE). — [Masrour et al. 2025](https://arxiv.org/html/2501.03437v1)
  - TPR 0% on o1-Pro text. — [Russell et al. 2025](https://arxiv.org/html/2501.15654)

**SICO (Lu et al., TMLR 2024): prompt-only, no fine-tuning**
- Substitution-based In-Context example Optimization automatically builds a reusable prompt using only 40 human-written examples and a limited number of LLM calls.
- It enables GPT-3.5 to evade six detectors, lowering AUC by 0.5 on average, and outperforms paraphraser baselines on three tasks.
- Human evaluation: human-level readability and task completion.
- The authors propose an ensemble defense.
- Source: [Lu et al., arXiv 2305.10847](https://arxiv.org/abs/2305.10847); [TMLR 2024 listing](https://mlanthology.org/tmlr/2024/lu2024tmlr-large/)

**AuthorMist (David & Gervais, UCL; arXiv 2503.08716, March 2025)**
- **Training.**
  - Qwen2.5-3B-Instruct trained with GRPO and a KL penalty (β = 0.001).
  - Reward R = 1 − mean P(AI) from a detector API.
  - Six separate models, each trained against one detector: GPTZero, Winston.ai, Sapling, Originality.ai, HelloSimpleAI, OpenAI-RoBERTa.
  - Cost: about 16 H100 GPU-hours per model.
  - Data: 10K CheckGPT abstracts, with AI versions from GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5.
  - Source: [AuthorMist](https://arxiv.org/html/2503.08716v1)
- **Attack success rate (evaluated on 300 XSum samples).**
  - Same-detector ASR is mostly ≥94% (GPTZero-trained: 92.33%).
  - Mean ASR across all six detectors: Originality-trained 95.17%, Winston 83.67%, GPTZero 72.83%, Sapling 71.11%, HelloSimpleAI 41.00%, OpenAI 32.33%.
  - The abstract's 78.6-96.2% range does not map cleanly onto the table.
  - Source: [AuthorMist](https://arxiv.org/html/2503.08716v1)
- **AUROC and comparisons.**
  - Mean AUROC 0.86 → 0.49 (Originality-trained model).
  - An SFT-only ablation reached AUROC 0.68 and ASR 65%, showing the RL step matters.
  - The paper says DIPPER had only "moderate success."
  - Source: [AuthorMist](https://arxiv.org/html/2503.08716v1)
- **Costs.**
  - E5 cosine similarity: median above 0.94 (about 0.945-0.975).
  - GPT-2 perplexity rose from about 25-40 (AI text) to as high as 300-400 for some models, compared with about 50-75 for human text. That suggests unnatural text.
  - The authors note semantic drift on technical text.
  - No held-out detector was tested.
  - Source: [AuthorMist](https://arxiv.org/html/2503.08716v1)

**StealthRL (Ranganath & Ramesh, UCSD; arXiv 2602.08934, February 2026)**
- **Training.**
  - Qwen3-4B-Instruct-2507 with LoRA, trained with GRPO.
  - Reward R = R_det + 0.1·R_sem, with R_det = 1 − (0.6·p_RoBERTa + 0.4·p_FastDetectGPT) and R_sem from E5 similarity.
  - Black-box access to detector scores only. Trained on 10K MAGE samples.
  - Source: [StealthRL v1](https://arxiv.org/html/2602.08934v1)
- **Results on MAGE test data (1K human / 1K AI, 1% FPR).**
  - RoBERTa-OpenAI: AUROC 0.829 → 0.693, TPR 0.225 → 0.002.
  - Fast-DetectGPT: AUROC 0.671 → 0.071, TPR 0.396 → 0.000.
  - Binoculars, held out from training: AUROC 0.716 → 0.041, TPR 0.410 → 0.001.
  - Mean TPR 0.344 → 0.001; ASR 99.9%.
  - Source: [StealthRL v1](https://arxiv.org/html/2602.08934v1)
- **Baseline mean TPR at 1% FPR.**
  - Plain LLM paraphrase: 0.079.
  - Adversarial Paraphrasing: 0.079.
  - AuthorMist: 0.037.
  - Homoglyph substitution (SilverSpeak): 0.001.
  - Source: [StealthRL v1](https://arxiv.org/html/2602.08934v1)
- **Quality cost, judged by gpt-5-nano.**
  - StealthRL: E5 similarity 0.896, quality 2.59/5, similarity 2.67/5, perplexity 48.7.
  - Plain paraphrase: 0.960, 4.01, 4.05, 21.3.
  - Adversarial Paraphrasing: 0.976, 4.03, 4.03, 18.9.
  - Homoglyph: 0.899, quality 2.01, perplexity 152.4.
  - Source: [StealthRL v1](https://arxiv.org/html/2602.08934v1)
- **v2 revision.** Evaluated on the full MAGE pool, adding the MAGE detector: near-zero detection on 3 of 4 detectors, mean TPR at 1% FPR 0.024. — [StealthRL v2 abstract via search](https://arxiv.org/pdf/2602.08934v2)
- **Caveats.** These are the authors' own numbers on one English benchmark, with no commercial detectors and no watermarks. Detector versions are given only as Hugging Face IDs. — [StealthRL v1](https://arxiv.org/html/2602.08934v1)

**Adversarial Paraphrasing (Cheng, Sadasivan, Saberi, Saha, Feizi; NeurIPS 2025): training-free, detector-guided decoding**
- **Mechanism.**
  - LLaMA-3-8B-Instruct paraphrases with a system prompt to rephrase and change style.
  - At each decoding step, the top-p (0.99) / top-k (50) candidate tokens are each appended and scored by a guidance detector, and the token with the lowest AI score is kept (a greedy, depth-1 search).
  - Guidance detectors tried: OpenAI-RoBERTa-Large/Base, MAGE, RADAR.
  - Source: [Cheng et al. 2025](https://arxiv.org/html/2506.07001v2); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116833)
- **TPR at 1% FPR: unattacked → simple paraphrase → Adversarial Paraphrasing (RoBERTa-Large guidance).**

  | Detector | Unattacked | Simple paraphrase | Adversarial Paraphrasing |
  |---|---|---|---|
  | RoBERTa-Large | 0.163 | 0.096 | 0.000 |
  | MAGE | 0.768 | 0.616 | 0.142 |
  | RADAR | 0.124 | 0.140 | 0.044 |
  | KGW watermark | 1.000 | 0.295 | 0.131 |
  | Unigram watermark | 0.999 | 0.609 | 0.294 |
  | Fast-DetectGPT | 0.323 | 0.326 | 0.003 |
  | GLTR | 0.174 | 0.049 | 0.001 |

  Source: [Cheng et al. 2025](https://arxiv.org/html/2506.07001v2)
- **Transfer.**
  - Average relative TPR reduction across detectors was 80.75-87.88%, depending on the guidance detector, versus 30.27% for simple paraphrasing.
  - Simple paraphrasing *increased* TPR on RADAR (+8.57%) and Fast-DetectGPT (+15.03%).
  - The authors' explanation: strong detectors converge on a shared model of "human text," so evading one evades others.
  - Source: [Cheng et al. 2025](https://arxiv.org/html/2506.07001v2)
- **Costs.**
  - GPT-4o judge: 4.48 vs 4.75 for simple paraphrase (the authors call the difference not significant); about 87% of outputs rated 4-5.
  - SBERT similarity: 0.81 vs 0.86.
  - Perplexity: 14.3-17.1, close to human text at 15.0 (simple paraphrase is 9.3, i.e., *more* predictable than human text).
  - Latency: 8.6-16.7 s vs 7.2 s per sample.
  - Failure: about 7.4% still flagged and about 4.15% backfired in transfer.
  - Source: [Cheng et al. 2025](https://arxiv.org/html/2506.07001v2)

**DPO-trained evaders and style-based detection (Rivera Soto, Chen, Andrews; arXiv 2505.14608, v2 September 2025)**
- **DPO against Fast-DetectGPT** (following Nicks et al. 2024), AUROC listed as Fast-DetectGPT / Binoculars / StyleDetect:
  - Mistral-7B: 0.72 / 0.70 / 0.96 → 0.18 / 0.17 / 0.95.
  - Mistral-Nemo: 0.75 / 0.79 / 0.97 → 0.37 / 0.33 / 0.96.
  - The style-based detector stays at about 0.95.
  - Source: [Rivera Soto et al. 2025](https://arxiv.org/html/2505.14608v2)
- **DPO against StyleDetect** did not reduce StyleDetect. The authors attribute this to the diversity of human style making the objective conflicting. — [Rivera Soto et al. 2025](https://arxiv.org/html/2505.14608v2)
- **Their style-aware paraphraser.**
  - Mistral-7B with LoRA, instruction-tuned to imitate a target author from 16 exemplars, refined with DPO, and applied iteratively.
  - It is "universally effective" against all 9 detectors tested, including style-based ones, *for a single sample*.
  - Detectability returns with multiple samples per source: about 5 samples for the DPO-FastDetectGPT model; about 50 (Reddit) or about 5 (Blogs) for their method.
  - Cost: edit distance 199 characters, semantic similarity 0.86, versus 0.93 for plain paraphrase and 0.84 for DIPPER.
  - Source: [Rivera Soto et al. 2025](https://arxiv.org/html/2505.14608v2)

**Commercial humanizer audit and a humanizer-robust detector (DAMAGE; Masrour, Emi, Spero, Pangram Labs; GenAIDetect at COLING 2025)**
- **Humanizers tested and quality tiers.**
  - L1 (best): DIPPER, GPTInf, Grammarly, HumanizeAI.pro, Quillbot, Semihuman, StealthGPT, Twixify.
  - L2: AIHumanizer.com, BypassGPT, StealthWriter, Surfer SEO.
  - L3 (worst): Ghost AI, Hix Bypass, HumanizeAI.io, Humbot, Phrasly, Undetectable AI, WriteHuman.
  - The paper says 19 tools; the table lists 18.
  - Source: [Masrour et al. 2025](https://arxiv.org/html/2501.03437v1)
- **TPR at 5% FPR on humanized academic text.**
  - GPTZero 60.04%, Binoculars 28.23%, RADAR 5.05%.
  - Pangram baseline (no humanizer training data): 95.74%. DAMAGE (humanizer-augmented): 98.26%.
  - At default thresholds: GPTZero 34.53%, Binoculars 29.73%.
  - Source: [Masrour et al. 2025](https://arxiv.org/html/2501.03437v1)
- **Adaptive attack.** GPT-4o fine-tuned on humanizer outputs that had evaded DAMAGE was still caught 100% of the time at 5% FPR (93.2% at the 0.5 threshold), and its fluency fell to the L2 level. — [Masrour et al. 2025](https://arxiv.org/html/2501.03437v1)
- **Humanizer behaviors observed.**
  - Synonym swapping, heavier sentence rewriting, adding or deleting content, shifting the reading level, and injecting typos or punctuation errors.
  - Low-tier tools insert nonsense, including fake citations and runs of "?".
  - LLM-based humanizers were jailbreakable, exposing their system prompts.
  - Source: [Masrour et al. 2025](https://arxiv.org/html/2501.03437v1)
- **Fluency.** GPT-4o fluency win rates against the original were 26.0% (L1), 14.67% (L2), and 2.67% (L3). Every humanizer degraded quality. — [Masrour et al. 2025](https://arxiv.org/html/2501.03437v1)
- **Conflict of interest.** These are the detector vendor's own numbers.

### Inferences
- **Optimization beats generic paraphrasing.** Generic paraphrasing (DIPPER, plain LLM rewriting) no longer reliably beats modern detectors and can even *raise* detection. Attacks optimized against a detector signal (RL, DPO, or guided decoding) do beat open-source detectors, and transfer across detector families.
- **The open question is commercial detectors trained on humanizer data.** No 2025-2026 academic attack paper reports an evaluation of RL- or guided-decoding attacks against current commercial detectors such as Pangram, GPTZero 2025+, or Originality 3.x. AuthorMist used commercial APIs as reward but reported no AUROC tied to a dated version.
- **The quality cost is visible.** StealthRL's 2.59/5 judge score and AuthorMist's perplexity of 300-400 suggest the strongest evasion produces text that is detectably odd by other signals.

### Gaps
- **"Raidar" (Mao et al., ICLR 2024).** This is a *detector* based on rewriting edit distance ("LLMs edit AI text less"), not an attack. I did not fetch its robustness numbers.
- **"TOBLEND"** (prompted in the brief): I found no reliable source and could not verify that the paper exists.
- **Evasive soft prompts (Kumarage et al., arXiv 2310.05095).** Identified, but I did not get numbers.
- **"Humanizing machine text" papers in general.** No papers beyond those above were confirmed with numbers.

## 4. Character-level attacks (homoglyphs, zero-width characters, whitespace, misspellings) and benchmark robustness (RAID, DetectRL)

### Takeaway
Character-level attacks cost nothing in meaning and produce the largest and most erratic swings. In RAID, homoglyphs cut Originality from 85.0% to 9.3% and RoBERTa-ChatGPT to 0.0%, and SilverSpeak cut average MCC from 0.64 to −0.01. Zero-width spaces flip detectors to all-AI or all-human. These attacks are trivially defeated by Unicode normalization, and GPTZero's RAID results suggest it already normalizes inputs.

### Cited Findings
- **RAID benchmark (Dugan et al., ACL 2024).** Over 6M generations from 11 generators across 8 domains, 11 attacks, and 4 decoding strategies; 12 detectors evaluated. — [RAID](https://arxiv.org/abs/2405.07940)
- **RAID's 11 attacks.**
  - Alternative spelling (British).
  - Article deletion.
  - Add paragraph breaks.
  - Upper-lower case swap.
  - Zero-width space (U+200B inserted every other character).
  - Whitespace between characters.
  - Homoglyph.
  - Number digit shuffle.
  - Misspelling.
  - Paraphrase (fine-tuned T5-11B, i.e., DIPPER).
  - Synonym swap (BERT candidates).
  - Source: [RAID](https://arxiv.org/html/2405.07940v2)
- **RAID Table 16: accuracy at 5% FPR, unattacked → attacked.**
  - **Homoglyph:**
    - RoBERTa-B-GPT2 59.1 → 7.6; RoBERTa-B-ChatGPT 44.8 → 0.0; Originality 85.0 → 9.3; Winston 71.0 → 26.3; Binoculars 79.6 → 37.7; GLTR 62.6 → 24.3; Fast-DetectGPT 73.6 → 51.4.
    - GPTZero 66.5 → 66.2 (essentially unaffected).
    - ZeroGPT 65.5 → 82.4 (rose).
  - **Zero-width space:**
    - Originality 85.0 → 4.9; RoBERTa-B-ChatGPT 44.8 → 0.0; Winston 71.0 → 25.0.
    - Rose for RoBERTa-GPT2 (99.9), Binoculars (99.1), Fast-DetectGPT (98.9), and GLTR (97.2).
    - The authors note it drives detectors to all-positive or all-negative labels.
  - **Whitespace:** RoBERTa-B-ChatGPT 44.8 → 0.1; LLMDet 35.0 → 4.4; Winston 71.0 → 46.8.
  - **Synonym:** Binoculars 79.6 → 43.5; Fast-DetectGPT 73.6 → 34.0; ZeroGPT 65.5 → 18.8; GLTR 62.6 → 31.2. But Originality rose 85.0 → 96.5 and RoBERTa-L-GPT2 rose 56.7 → 79.4.
  - **Paraphrase (DIPPER):** ZeroGPT 65.5 → 46.7; Winston 71.0 → 52.6; GLTR 62.6 → 47.2. But Originality rose 85.0 → 96.7 and Binoculars rose 79.6 → 80.3.
  - **Misspelling:** Mostly small effects, e.g., Originality 85.0 → 78.6 and Binoculars 79.6 → 78.0.
  - **Article deletion:** Originality 85.0 → 71.4; RoBERTa-L-GPT2 56.7 → 33.2.
  - **GPTZero overall:** its worst drop across all 11 attacks was the upper-lower swap (66.5 → 56.5).
  - Source: [RAID](https://arxiv.org/html/2405.07940v2)
- **RAID's own framing.** The authors do not rank attacks overall. They say effects are detector-specific and that some attacks *improve* accuracy. — [RAID](https://arxiv.org/html/2405.07940v2)
- **SilverSpeak (Creo & Pudasaini, GenAIDetect 2025, arXiv 2406.11239).**
  - Method: Latin → Cyrillic homoglyph swaps, either random (5-20% of characters) or greedy (every character that has a homoglyph).
  - Tested against 7 detectors (ArguGPT, Binoculars, DetectGPT, Fast-DetectGPT, Ghostbuster, OpenAI detector, watermarking) on 5 datasets.
  - Average MCC fell from 0.64 to −0.01.
  - Mechanism: homoglyphs break tokenization, which inflates token counts and shifts perplexity.
  - Source: [SilverSpeak abstract](https://arxiv.org/abs/2406.11239); [ACL Anthology](https://preview.aclanthology.org/setup/2025.genaidetect-1.1); mechanism from [secondary review](https://www.themoonlight.io/en/review/silverspeak-evading-ai-generated-text-detectors-using-homoglyphs)
- **Homoglyph cost in StealthRL's evaluation.** Mean TPR 0.001, but judge quality 2.01/5 and perplexity 152. — [StealthRL](https://arxiv.org/html/2602.08934v1)
- **Early precedent.** "Evade ChatGPT Detectors via A Single Space" (arXiv 2307.02599, 2023) showed whitespace-insertion evasion; I did not get its numbers. — [arXiv 2307.02599](https://arxiv.org/pdf/2307.02599)
- **DetectRL (Wu et al., NeurIPS 2024 Datasets & Benchmarks).**
  - Attack categories:
    - Prompt attacks: few-shot, SICO-style ICO.
    - Paraphrase attacks: DIPPER, back-translation, LLM polishing.
    - Perturbation attacks: DeepWordBug (character-level), TextFooler (word-level), TextBugger (sentence-level).
    - Data mixing: about 1/4 of the text replaced with human sentences.
  - Zero-shot detectors, average AUROC: direct 77.67, prompt 75.70, paraphrase 62.00, perturbation 39.24, mixing 59.50.
  - Binoculars was the most robust zero-shot detector (88.53 average; 76.89 under perturbation). DetectGPT averaged 36.19.
  - Supervised RoBERTa/XLM-R detectors stayed at 96-99.9 AUROC under every attack.
  - Source: [DetectRL](https://arxiv.org/html/2410.23746v3)

### Inferences
- **Character attacks matter in practice even though they are cheap to stop.** Normalization (NFKC, confusable mapping, stripping zero-width and format characters) neutralizes them. GPTZero's flat RAID results under homoglyph, zero-width, and whitespace attacks are consistent with input normalization. Detectors without normalization (Originality and Winston in the 2024 RAID run) collapsed.
- **Some character attacks are themselves a signal.** Zero-width spaces and homoglyphs push some detectors toward always saying "AI" (ZeroGPT under homoglyphs; Binoculars and Fast-DetectGPT under zero-width). Their presence is a strong anomaly flag for a detector to use.
- **DetectRL's supervised scores are probably optimistic.** The supervised detectors were most likely trained in-distribution on DetectRL's own attack data, which would explain the 96-99 AUROC. This is my inference; I did not verify the training split.

### Gaps
- **RAID leaderboard.** I did not fetch the live leaderboard, which may include 2025 detectors such as Pangram and newer GPTZero versions.
- **MAGE and M4 attack results.** Not fetched. MAGE (Li et al., ACL 2024) has an out-of-distribution paraphrase test set. M4 is multilingual and mainly about cross-domain and cross-generator shift.

## 5. Which transformations survive modern detectors best, and at what quality cost

### Takeaway
Ranked by evidence, from most to least effective:
1. **Detector-optimized rewriting.** RL, DPO, or guided decoding produce lexical choices that look human under the detector's own statistic, and this survives across open detectors.
2. **Character-level obfuscation.** Highly effective unless the input is normalized.
3. **Data mixing and partial human editing.**
4. **Generic paraphrasing.** Weakening against current commercial and supervised detectors.
5. **Prompt-only style changes.** Weakest.

Style- and discourse-level signals (StyleDetect, expert human readers) were the most resilient to every class of attack.

### Cited Findings
- **Perturbation vs paraphrase vs prompting, zero-shot detectors.** Perturbation drops average AUROC from 77.67 to 39.24, paraphrase to 62.00, and prompting only to 75.70. — [DetectRL](https://arxiv.org/html/2410.23746v3)
- **Word-level synonym swaps hit perplexity-based zero-shot detectors hard but not trained ones.** Binoculars −36.1 and Fast-DetectGPT −39.6, but Originality and RoBERTa-GPT2 *improved*. — [RAID](https://arxiv.org/html/2405.07940v2)
- **Generic paraphrasing is now partly a detectable signature.**
  - DIPPER raised Originality to 96.7% in RAID. — [RAID](https://arxiv.org/html/2405.07940v2)
  - Simple LLM paraphrase raised RADAR and Fast-DetectGPT TPR. — [Cheng et al. 2025](https://arxiv.org/html/2506.07001v2)
- **Token-level vs style-level optimization.**
  - DPO against Fast-DetectGPT drops token-statistic detectors below chance, while StyleDetect stays at about 0.95 AUROC. — [Rivera Soto et al. 2025](https://arxiv.org/html/2505.14608v2)
  - Explicit author-style imitation from 16 exemplars defeats style detectors on a single sample, but not across about 5-50 samples. — [Rivera Soto et al. 2025](https://arxiv.org/html/2505.14608v2)
- **Semantic-similarity costs.**

  | Method | Similarity | Other quality signal | Source |
  |---|---|---|---|
  | DIPPER L60/O60 | P-SP 0.88-0.96 | — | [Krishna](https://arxiv.org/pdf/2303.13408) |
  | Adversarial Paraphrasing | SBERT 0.81 (simple paraphrase 0.86) | — | [Cheng](https://arxiv.org/html/2506.07001v2) |
  | StealthRL | E5 0.896 (plain paraphrase 0.960) | Judge quality 2.59 vs 4.01 | [StealthRL](https://arxiv.org/html/2602.08934v1) |
  | AuthorMist | E5 median > 0.94 | GPT-2 perplexity up to 300-400 | [AuthorMist](https://arxiv.org/html/2503.08716v1) |
  | Style paraphraser | Semantic similarity 0.86, 199 characters edited | — | [Rivera Soto](https://arxiv.org/html/2505.14608v2) |
  | Recursive paraphrase pp5 | 70-83% rated high on content preservation | Perplexity +3 to +5 | [Sadasivan](https://arxiv.org/html/2303.11156v4) |
  | Commercial humanizers | — | Fluency win rate vs original 2.7-26% | [DAMAGE](https://arxiv.org/html/2501.03437v1) |

- **Perplexity overshoot.** Simple paraphrasing makes text *more* predictable than human text (perplexity 9.3 vs 15.0), while detector-guided paraphrasing lands at 14-17, about human-like. Hitting the human perplexity band is part of why guided decoding works. — [Cheng et al. 2025](https://arxiv.org/html/2506.07001v2)

### Inferences
- **Lexical predictability is the weak point; style and discourse are more robust.** Modern evasion succeeds mainly by moving token-probability statistics (perplexity, cross-perplexity, curvature) into the human range. Detectors that rely on those statistics alone (Fast-DetectGPT, Binoculars, GLTR) are the most attackable. Style, discourse-structure, and content features (expert clues such as formulaic structure, "Dr./Prof." naming, and quote patterns) remain more robust.
- **Detector-side countermeasures this suggests:**
  - Normalize Unicode.
  - Flag perplexity that is anomalously high or erratic, as produced by homoglyphs and RL overshoot.
  - Train on humanizer outputs, as DAMAGE did.
  - Add style and discourse features.
  - Aggregate over multiple documents where possible.

### Gaps
- No paper I found isolates discourse-level edits (reordering arguments, restructuring paragraphs) as a separate attack with numbers. DIPPER's O (order) knob is the closest: adding O60 to L60 cut DetectGPT from 8.7 to 4.6 and GPTZero from 7.1 to 1.2 on GPT2.
- I found no human-evaluation study comparing BERTScore across all attack families on one benchmark.

## 6. Prompt-only methods with off-the-shelf LLMs

### Takeaway
Prompt-only evasion is the weakest attack class against modern detectors, but it is not useless:
- Optimized in-context human examples (SICO) cut AUC by about 0.5 against 2023 detectors.
- An expert-guidebook prompt to o1-Pro badly hurt zero-shot detectors (Binoculars accuracy 73.3% → 6.7%) and GPTZero (76.7% → 46.7%), but barely moved Pangram (100% → 90-97%) or a majority vote of expert humans (100% → 100%).
- Detector-in-the-loop prompting and decoding (Adversarial Paraphrasing) is the strongest training-free option.

### Cited Findings
- **SICO.** 40 human examples plus iterative substitution and in-context optimization; AUC −0.5 on average across 6 detectors (2023-era) with GPT-3.5; human-level readability. — [Lu et al.](https://arxiv.org/abs/2305.10847)
- **DetectRL prompt attacks** (few-shot plus ICO) were the *least* effective attack category: zero-shot average AUROC 77.67 → 75.70, and supervised detectors were unaffected (99.09). — [DetectRL](https://arxiv.org/html/2410.23746v3)
- **Russell et al. 2025 (arXiv 2501.15654), "People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text."**
  - **Prompt design.** Expert-derived clues (vocabulary, tone, structure) were compiled into a guidebook, and o1-Pro was prompted to write articles that evade a detector following it.
  - **TPR (FPR), plain → humanized:**
    - Pangram-Humanizers: 100 (0) → 96.7 (10).
    - Pangram: 100 (0) → 90.0 (6.7).
    - GPTZero: 76.7 → 46.7.
    - Fast-DetectGPT: 86.7 → 23.3.
    - Binoculars accuracy: 73.3 → 6.67.
    - RADAR: 0 → 0.
  - **Human experts.** Majority vote stayed at 100% TPR / 0% FPR, but confidence fell. Experts still relied on vocabulary (cited 42.3% vs 57.1% of the time), sentence structure, originality, quotes, and titles.
  - **Sample size.** Small: about 30 articles per condition, inferred from the 3.3% increments. Treat as indicative.
  - Source: [Russell et al. 2025](https://arxiv.org/html/2501.15654)
- **Imperfection instructions.** A CLEF 2025 shared-task notebook reported that instructing the model to add human-like imperfections (grammar errors, awkward phrasing) mattered more for evasion than re-translation. This is a single-team workshop report. — [CEUR-WS Vol-4038 paper 116](https://ceur-ws.org/Vol-4038/paper_116.pdf)
- **Plain prompted rewrites vs detector feedback.** Plain GPT-4o-mini prompting or paraphrasing was a baseline in Rivera Soto et al. (edit distance 134, similarity 0.90) and was outperformed by detector-optimized methods. — [Rivera Soto et al. 2025](https://arxiv.org/html/2505.14608v2)
- **Iterative refinement with detector feedback.** Adversarial Paraphrasing is the training-free form: an off-the-shelf LLaMA-3-8B with a detector choosing tokens, giving an 87.88% average relative TPR reduction. — [Cheng et al. 2025](https://arxiv.org/html/2506.07001v2)
  - Its output-level analogue, best-of-n resampling scored by a detector, is in Sadasivan et al.: best-of-pp5 brought watermark TPR to 4.0%. — [Sadasivan et al.](https://arxiv.org/html/2303.11156v4)

### Inferences
- **Prompting changes surface cues but not deeper ones.** Persona or style prompts and "avoid AI vocabulary" guidebooks change vocabulary and tone, which defeats zero-shot statistical detectors. They do not remove the deeper structural and content signature that supervised, humanizer-trained classifiers (Pangram) and expert readers pick up.
- **Effectiveness ordering for prompt-only methods.** Detector-feedback loops (resample or select by detector score) > optimized in-context human exemplars (SICO) > guidebook or persona instructions > generic "write like a human."

### Gaps
- I found no controlled 2025-2026 study isolating persona prompting vs in-context human examples vs imperfection instructions against current commercial detectors with dated versions.
- I did not get numbers for Kumarage et al.'s evasive soft prompts (arXiv 2310.05095), which require white-box access and so are not prompt-only in the practical sense.

## Older-detector caveat (summary flag for the writer)
- **Historical only:**
  - OpenAI's AI text classifier (retired July 2023).
  - DetectGPT (superseded by Fast-DetectGPT and Binoculars).
  - GPTZero numbers from early 2023 in DIPPER.
  - Numbers from "the six detectors" in SICO (2023).
- **RAID (2024) commercial results.** GPTZero, Originality, Winston, and ZeroGPT as of roughly early 2024; vendors have updated since.
- **Most current evidence (2025-2026):**
  - Adversarial Paraphrasing (NeurIPS 2025): open detectors only.
  - StealthRL (2026): open detectors only.
  - Rivera Soto et al. (2025): open plus style detectors.
  - DAMAGE (2025): commercial humanizers vs GPTZero, Binoculars, RADAR, Pangram; vendor-authored.
  - Russell et al. (2025): Pangram, GPTZero, Binoculars, Fast-DetectGPT, RADAR; small sample.
