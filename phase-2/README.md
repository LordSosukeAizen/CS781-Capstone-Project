# Phase 2: Legal Named Entity Recognition (L-NER)

Each Phase-1 model is adapted to predict **character spans** `[start_char, end_char, label]` for **12 legal entity types** (for example `APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`, `JUDGE`, `COURT`, `AUTH`, `PREC`, `STAT`, `CASENO`, `DATE`, `WIT`).

**Metric:** strict Macro-F1. A prediction counts only if `(doc_id, start, end, label)` exactly matches a gold entity, so an entity with a wrong boundary is wrong.

| Notebook | Backbone | Best dev strict Macro-F1 |
|---|---|---|
| [encoder-decoder-phase-2.ipynb](encoder-decoder-phase-2.ipynb) | Encoder–decoder | 0.231 (enc + dec fusion, LoRMA⁺) |
| [gqla-phase-2.ipynb](gqla-phase-2.ipynb) | GQLA | 0.371 (partial FT) |
| [mamba-phase-2.ipynb](mamba-phase-2.ipynb) | Mamba | **0.494** (Houlsby adapters) |
| [transformer-xl-gqa-phase-2.ipynb](transformer-xl-gqa-phase-2.ipynb) | Transformer-XL + GQA | 0.253 (full FT) |

---

## 1. Data
`Exploration-Lab/CS-781-Capstone` → `Phase_2_LNER/`
- `Train/fold1-text.json`, `fold1-gold.json`: used as the **training** split
- `Train/fold2-text.json`, `fold2-gold.json`: used as the **local dev** split (the brief gives no separate dev set)
- `Test/fold3-text.json`: private test fold, predicted and submitted
- `Train/labels.txt`: label inventory

## 2. Shared pipeline
1. **Char ↔ word ↔ BPE alignment.** The Phase-1 tokenizer has no offset tracking, so labels are aligned at word level first (`WORD_RE`, the regex used to train the tokenizer). Each word keeps its `(start, end)` in the *raw* text. For encoding, a word is normalised exactly as in pretraining, so the model sees familiar ids while offsets still point into the original string.
2. **BIO tags** per word, expanded to sub-tokens. Decoding joins consecutive tagged words into `(first_word.start, last_word.end)`.
3. **Alignment ceiling.** A gold boundary that falls *inside* a word can never be produced with word-level BIO. The best strict Macro-F1 reachable is **0.9876 (train) / 0.9877 (dev)**. The weakest labels are `PREC` (0.93), `COURT` and `APP`.
4. **Overlapping sliding windows** (512 tokens). Probabilities from overlapping windows are merged, with tokens near the centre of a window weighted more, so no entity is cut at a chunk edge.
5. **BIO-constrained Viterbi decoding**, compared with plain argmax as a span-decoding ablation.
6. **Submission**: `prediction.json` at the root of `submission.zip`. It is validated first: every test id is present, there are no unknown ids, offsets are integers inside the text, and labels are valid.

## 3. Adaptation techniques
The brief requires at least 5 techniques plus a mandatory KD run, with **LoRMA compulsory**. Every experiment starts from the Phase-1 checkpoint.

| Technique | Notes |
|---|---|
| Full fine-tuning | Also the KD teacher |
| Partial / layer-wise FT | Bottom layers and embeddings frozen |
| LoRA | `y = W₀x + (α/r)·B·A·x`, with `B = 0` at init |
| **LoRMA⁺** (Bihany, Patel & Modi, 2025) | Multiplicative: `W = (I + (α/r)·B·A)·W₀`, computed as `h = W₀x; y = h + (α/r)·B(A·h)`. `B = 0`, so the layer equals Phase 1 at init |
| Houlsby adapters | Bottleneck 64 |
| BitFit | Bias-only tuning |
| **Knowledge distillation** | Frozen FT teacher → smaller student initialised from Phase-1 layers |
| Transformer-XL extras | Pfeiffer adapters, prefix tuning (through the XL memory slot), prompt tuning, span-extraction head |

---

## 4. Results (dev = fold2, strict Macro-F1)

### Encoder–decoder (`LegalSeq2SeqLM`)
The bidirectional **encoder** gets a token-classification head. Experiment 7 also keeps the decoder: the decoder reads `[<s>] + window[:-1]` with teacher forcing, and the head classifies `[enc_hᵢ ; dec_hᵢ]`.

| Exp | Technique | Trainable | Dev F1 (Viterbi) | Argmax | Train time |
|---|---|---|---|---|---|
| exp1 | Full FT | 54.5M (100%) | 0.048 | 0.043 | 3.7 min |
| exp2 | Partial FT (bottom 5/9 + emb frozen) | 16.8M (30.8%) | 0.057 | 0.047 | 3.5 min |
| exp3 | LoRA r=16 (q,k,v,o) | 0.61M (1.1%) | 0.070 | 0.054 | 2.2 min |
| exp4 | LoRMA⁺ r=16 (q,k,v,o) | 0.61M (1.1%) | 0.068 | 0.053 | 2.1 min |
| exp5 | Houlsby adapters | 1.21M (2.2%) | 0.075 | 0.054 | 2.1 min |
| exp6 | KD: 9-layer teacher → 5-layer student | 37.8M | 0.061 | 0.052 | 3.3 min |
| **exp7** | **LoRMA⁺ on encoder + decoder, fusion head** | 1.82M (1.8%) | **0.231** | 0.140 | 10 min |

Encoder-only heads stay at 0.05–0.07. Adding the decoder state triples the score, so for this checkpoint the decoder adds real information for NER.

### GQLA
| Exp | Technique | Trainable | Dev F1 |
|---|---|---|---|
| exp2 | **Partial / layer-wise FT** | 30.7M (50%) | **0.371** |
| exp1 | Full FT | 61.3M (100%) | 0.282 |
| exp3 | LoRA r=8 | 0.45M (0.7%) | 0.210 |
| exp4 | "LORMA" r=8 (best-effort gated-LoRA reconstruction) | 0.45M (0.7%) | 0.143 |
| exp6 | KD (randomly initialised small student) | 14.3M | 0.029 |

The Mamba notebook lists what went wrong here and fixes it:
- causal backbone with the tag read from a word's **first** sub-token (no right context)
- "LORMA" was a gated LoRA with `B ≠ 0`, not the paper's formula
- the KD student was **randomly initialised**, which also breaks "start from Phase 1"
- no LR schedule, and too few epochs
- dev scored against BIO-reconstructed spans instead of gold spans
- the submission cell always rebuilt a partial-FT model

The submitted zip is `submission_exp2_partial_ft.zip`.

### Mamba (20 blocks, d=512)
Default head: causal Mamba states → **BiGRU** → linear, which adds right context. Every sub-token is trained on its word's tag. Up to 10 epochs with warm-up and linear decay, head LR × 10, and early stopping on dev F1.

| Exp | Technique | Trainable | Dev F1 (Viterbi) | Argmax | Train time |
|---|---|---|---|---|---|
| **exp5** | **Houlsby adapters (b=64)** | 2.52M (3.5%) | **0.494** | 0.425 | 208 min |
| exp3 | LoRA r=16 (`in_proj`, `out_proj`) | 3.00M (4.1%) | 0.493 | 0.436 | 216 min |
| exp4 | LoRMA⁺ r=16 | 3.49M (4.8%) | 0.468 | 0.377 | 151 min |
| exp6 | KD: 20-block teacher → 10-block student (blocks 0,2,…,18) | 44.3M | 0.463 | 0.390 | 141 min |
| exp1 | Full FT | 70.7M (100%) | 0.429 | 0.342 | 109 min |
| exp2 | Partial FT (top 10/20 blocks) | 27.5M (39%) | 0.486 at epoch 6* | — | — |

\*exp2 reached 0.486 at epoch 6 but is missing from the final summary table, so the run did not finish or save its result.

Extras defined but not reported: `x_bitfit`, `x_ssm_only` (Mamba-specific Δ/A/D tuning), `x_linear_head` (causal-only ablation), `x_bidir` (reversed second pass).

**Error analysis (exp5 adapters):**
- Error counts: missed 929, spurious 616, boundary 496, wrong type 226, boundary + type 104
- Top confusions: `A.COUNSEL → R.COUNSEL` (61), `APP → PREC` (18), `R.COUNSEL → A.COUNSEL` (17), `PREC → COURT` (15)
- Recall by position in the document is 0.61–0.63 for the first half and drops to **0.54** in the last quarter, a long-document effect.

### Transformer-XL + GQA (76M, small checkpoint)
Each window is classified **without** carrying XL memory across windows, so one window's content cannot leak into another's boundary decisions. Ten experiment slots are implemented; the eight that ran are below.

| Exp | Technique | Trainable | Dev F1 | Precision | Recall |
|---|---|---|---|---|---|
| **E1** | **Full FT** | 76.0M | **0.253** (rerun: 0.226) | 0.215 | 0.330 |
| E4 | LoRMA (π and + variants) | 0.24M (0.3%) | 0.235 | 0.192 | 0.336 |
| E3 | LoRA | 0.18M (0.2%) | 0.198 | 0.161 | 0.278 |
| E6 | KD (smaller TXL student) | 11.2M | 0.188 | 0.190 | 0.204 |
| E2 | Partial FT (top 2 layers) | 12.7M (16.7%) | 0.188 | 0.158 | 0.257 |
| E5 | Houlsby adapter | 1.61M (2.1%) | 0.179 | 0.145 | 0.253 |
| E9 | BitFit | 0.06M (0.08%) | 0.073 | 0.072 | 0.081 |

All TXL runs use 512-token windows with 64-token overlap and a BIO head. Each run took 3.6–6.2 hours. Section 18 of the notebook adds a context-window study, and Section 19 a domain error analysis.

---

## 5. Observations
- **Mamba is clearly the best NER backbone** (0.49 vs at most 0.37 for the others). PEFT methods (adapters, LoRA) match or beat full fine-tuning while training only 3–5% of the weights.
- With the correct formula and `B = 0` init, **LoRMA⁺** sits 0.02–0.03 below LoRA on Mamba and slightly above it on Transformer-XL.
- **KD needs a Phase-1-initialised student**: 0.029 with a random student (GQLA) vs 0.463 with layer-dropped Phase-1 blocks (Mamba).
- **Viterbi decoding** beats argmax by 0.05–0.09 F1 on every model.
- All scores are far below the 0.988 alignment ceiling, so the main error source is modelling, not tokenisation.

## 6. How to run
1. Set the Drive paths to your Phase-1 checkpoint and tokenizer (`final_bpe_en.json`). The architecture classes are copied unchanged so `load_state_dict` matches strictly.
2. Add `HF_TOKEN` as a Colab secret.
3. Run sections 0–10 (setup, data, model, training loop), then each experiment cell. Each one saves its best weights and results to Drive, so after a disconnect you re-run 0–10 and continue with the next experiment.
4. The submission section builds one validated `submission.zip` per experiment.
