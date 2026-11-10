# Phase 4: Court Judgment Prediction (CJPE)

Binary outcome prediction (**0 / 1**) for long Indian court judgments, using each Phase-1 model.

**Metric:** Macro-F1. Each run reports it both at threshold 0.5 and at a dev-tuned threshold. The tuned number is optimistic because the threshold is fitted on the same dev data, so the **@0.5 number is the stricter one**.

| Notebook | Backbone | Best dev Macro-F1 @0.5 (tuned) |
|---|---|---|
| [encoder-decoder-phase-4.ipynb](encoder-decoder-phase-4.ipynb) | Encoder–decoder (encoder only) | 0.552 (0.563): frozen encoder + GRU |
| [gqla-phase-4.ipynb](gqla-phase-4.ipynb) | GQLA | **0.736 (0.745)**: full FT |
| [mamba-phase-4.ipynb](mamba-phase-4.ipynb) | Mamba | 0.584 (0.584): partial FT |
| [transformer-xl-phase-4.ipynb](transformer-xl-phase-4.ipynb) | Transformer-XL + GQA | 0.333: every run collapsed to one class |

---

## 1. Data
`Exploration-Lab/CS-781-Capstone` → `Phase_4_CJPE/` (`Train/train.json`, `Train/dev.json`, `Test/test.json`)

| Split | Docs |
|---|---|
| train | 37,387 (about 41% label 1, shorter documents) |
| dev | 3,505 in the encoder–decoder loader; 1,988 in the decoder-only notebooks (about 50/50) |
| test | 1,517 |

The decoder-only notebooks keep **1,000 dev docs as `dev_val`** for threshold and epoch selection. The rest of dev is added to training (repeated `DEV_REPEAT`×), because dev matches test's length profile better than train does.

## 2. Where is the outcome signal? (CPU lexical probe)
Section 4 of the decoder-only notebooks fits TF-IDF over BPE unigrams and hashed bigrams, plus logistic regression, **one region of the document at a time**. It is a diagnostic only and is never submitted.

| Region | F1 @0.5 | F1 tuned |
|---|---|---|
| middle | 0.543 | 0.564 |
| last 256 tokens | 0.588 | 0.617 |
| last 512 tokens | 0.584 | 0.631 |
| **last 1,024 tokens** | **0.620** | **0.655** |
| whole document | 0.619 | 0.644 |

These are the GQLA notebook's numbers; Mamba and TXL get 0.60–0.63. The cue lives in the **tail** of the judgment (for example "set aside"), so the decoder-only notebooks feed only the tail windows.

Frozen-feature probe (Phase-1 tail-512 features + logistic regression), macro-F1 @0.5: GQLA **0.553**, Mamba 0.539, TXL 0.536.

## 3. Models

### Encoder–decoder: chunk → pooling → document label
`document tokens → windows → Phase-1 encoder → mean-pool per window → [+ position features] → aggregator → 2-way logits`

- **Views**: `head`, `tail`, `headtail`, `sliding`, `sliding_last`, `sliding_first`, `sliding_headtail`
- **Aggregators**: `mean`, `max`, `attn`, `gru` (BiGRU + attention)
- **Methods**: frozen, full, partial, `gradual` (layer-wise unfreezing), LoRA, rsLoRA, LoRMA⁺, Houlsby adapters, IA³, `norm` (a BitFit analogue that trains only RMSNorm gains, because the model has no biases), prompt tuning, KD

### Decoder-only (GQLA, Mamba, TXL)
- Documents are trimmed to the **first 2,048 + last 6,144 tokens**.
- `view` lists windows: `T0` = last 512 tokens, `T1` = the window before it, `H0` = first window. The main runs use `T0,T1` concatenated.
- Each window is summarised as **[last-token ; masked-mean ; masked-max]**. Max-pooling fires on cue words anywhere in the window.
- Real mini-batches, deterministic windows, class-weighted CE, warm-up + cosine, AMP, a guard against non-finite losses, epoch selection on `dev_val`.
- **Techniques**: B1 full FT (also the KD teacher) · B2 partial FT (top 3) · B3 **LoRMA⁺** · B4 Houlsby adapters · B5 BitFit · B6 KD student (B6b: the same student without KD) · extras X1 gradual unfreezing, X2 LoRA, X3 rsLoRA, X4 IA³, X5 prompt, X6 prefix (TXL memory).

| | GQLA | Mamba | Transformer-XL |
|---|---|---|---|
| Backbone | 10 pre-LN blocks, d=512, decoupled RoPE | 20 Mamba blocks, d=512, no positional encoding | 8 post-LN blocks, d=768, relative attention, GQA 32q/8kv |
| Window feature | 1,536 dims | 1,536 dims | 2,304 dims |
| BitFit trains | FFN + LayerNorm biases | `conv1d.bias`, `dt_proj.bias` only | FFN + LayerNorm biases |
| KD student | bottom 5 blocks (52.5M) | smaller Mamba stack (43.7M) | blocks 0, 2, 5, 7 (51.4M) |
| Throughput (train) | 2.25 docs/s | 0.26 docs/s | 3.08 docs/s |

## 4. Results

### GQLA (`dev_val`, 1,000 docs)
| Exp | Technique | Trainable | F1 @0.5 | F1 tuned | Test positive rate |
|---|---|---|---|---|---|
| **B1** | **Full FT** | 87.7M | **0.736** | **0.745** | 0.50 |
| X1 | Gradual unfreezing | 70.9M | 0.701 | 0.714 | 0.46 |
| B4 | Houlsby adapters | 1.9M | 0.695 | 0.705 | 0.56 |
| B2 | Partial FT (top 3) | 21.7M | 0.691 | 0.694 | 0.46 |
| B6 | KD student | 52.5M | 0.664 | 0.682 | 0.61 |
| B5 | BitFit | 0.63M | 0.553 | 0.601 | 0.52 |
| B3 | LoRMA⁺ | 2.5M | 0.502 | 0.502 | 0.46 |
| B6b | Same student, no KD | 52.5M | 0.339 | 0.504 | 0.68 |
| X2 | LoRA | 2.2M | 0.339 (stopped early) | — | — |

Full FT clears the 0.7 target. KD is decisive here: the same student scores 0.664 with the teacher and 0.339 without it. LoRMA⁺ and LoRA barely move from chance on this task.

### Mamba
| Exp | Technique | Trainable | F1 @0.5 | Test positive rate |
|---|---|---|---|---|
| B2 | Partial FT (top 3) | 8.5M | **0.584** | 0.48 |
| B1 | Full FT | 70.1M | 0.406 | 0.05, nearly all class 0 |
| B4, B6 | Adapters, KD | — | started, no result saved | — |

Mamba is about 10× slower per document than GQLA (0.26 docs/s), which limited how many runs finished.

### Transformer-XL + GQA
Every experiment (B1–B6b, X1–X6) ends at **macro-F1 0.333 @0.5**, with a test positive rate of 0.0 or 1.0 (B1 and B2 are near 0.01–0.03). Training loss stays at ≈ ln 2 = 0.693: the model never fits the training set and predicts a single class. The tuned-threshold numbers (≤ 0.35) are also chance level. None of these submissions are useful.

### Encoder–decoder (full dev)
| Exp | Setup | F1 @0.5 | F1 tuned |
|---|---|---|---|
| **A0** | **Frozen encoder, sliding windows, GRU aggregator** | **0.552** | **0.563** |
| B6b | 4-layer student, no KD | 0.543 | 0.561 |
| A1 | LoRA, head view | 0.396 | 0.544 |
| A2 | LoRA, tail view | 0.484 | 0.484 |
| A3, A6, A7b, A8a, A9 | LoRA with other views or aggregators | 0.33–0.35 (single class) | 0.33–0.50 |
| B1, B2, B3, B5, B7e | Full, partial, gradual, adapter, prompt | **0.5125, identical** | 0.5673, identical |

**Known bug:** in this notebook, the Part-B runs produce byte-identical histories and scores. `BEST_REP` contained `method="frozen"` and was spread *after* `{"method": "full", ...}`, so every B-run silently trained the **frozen** model. The decoder-only notebooks fix this: `REP` holds only representation keys, and `run()` asserts that non-frozen methods have more trainable parameters than the head. **The encoder–decoder B-results should be re-run before they are reported.**

---

## 5. Observations
- The outcome signal is concentrated in the **last ~1,000 tokens** of a judgment.
- **GQLA full fine-tuning (0.736) is the best CJPE model.** It beats the lexical probe (0.620), so the fine-tuned Phase-1 model adds value beyond keyword cues.
- On this task PEFT lags full FT more than in other phases. Frozen features are close to chance, and the signal has to be learned into the backbone.

## 6. How to run
1. Sections 1–5: setup, data, tokenisation (cached), the optional CPU lexical probe, and loading the Phase-1 model.
2. Run `benchmark()` to measure docs/s, then set `docs_per_epoch`. Run `SMOKE_TEST = True` once.
3. Run **B1 first** (it is the KD teacher and is saved to disk), then B2–B6 and the extras. Finished experiments are skipped on re-run (`FORCE_RERUN=True` to redo).
4. Submit each `experiments/<name>/submission.zip`.
