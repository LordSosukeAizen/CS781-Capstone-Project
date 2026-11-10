# Phase 3: Rhetorical Role (RR) Prediction

Every **sentence** of a judgment is labelled with one of **13 rhetorical roles** (for example Facts, Issue, Argument, Precedent, Ratio, Ruling, Dissent, PrecedentOverruled, None). The data has two domains: **CL** and **IT**.

**Metric:** Macro-F1, averaged over the two domains. Two variants are reported:
- `mean_F1` / "union": sklearn's default, over labels present in gold ∪ predicted. This is the headline number.
- `mean_F1_strict13`: all 13 labels, so a label that never occurs in dev counts as 0.

| Notebook | Backbone | Best single model (mean F1) | With ensemble + decoding |
|---|---|---|---|
| [encoder-decoder-phase-3.ipynb](encoder-decoder-phase-3.ipynb) | Encoder–decoder (encoder only) | 0.424 (LoRMA⁺) | — |
| [gqla-phase-3.ipynb](gqla-phase-3.ipynb) | GQLA | **0.518** (Houlsby adapters) | **0.554** |
| [transformer-xl-gqa-phase-3.ipynb](transformer-xl-gqa-phase-3.ipynb) | Transformer-XL + GQA | 0.252 (frozen + position) | 0.251 |

There is no Mamba notebook for this phase.

---

## 1. Data
`Exploration-Lab/CS-781-Capstone` → `Phase_3_RR/` (`Train/{CL,IT}_{train,dev}.json`, `Test/{CL,IT}_test.json`, `Train/label_vocab.json`)

| Split | Docs | Sentences |
|---|---|---|
| CL train / dev / test | 40 / 5 / 5 | 10,593 / 1,396 / 1,339 |
| IT train / dev / test | 40 / 5 / 5 | 6,252 / 746 / 858 |

**No sentence is ever dropped.** Empty sentences become `"."`, so the number of predictions always equals the number of test sentences (2,197), which the submission format requires.

> Dev has only **5 documents per domain**, so differences of 1–2 points between methods are within noise (the notebooks estimate about ±0.05).

## 2. Model: hierarchical, document-aware classifier
Earlier versions classified each sentence in isolation from its last-token state and plateaued at Macro-F1 ≈ 0.25–0.27. All three notebooks now use one shared design:

```
sentence → Phase-1 backbone → pooled sentence vector
        → [+ relative-position features + domain embedding (CL / IT)]
        → 2-layer BiGRU / BiLSTM over all sentences of the document
        → 13-way label per sentence
```

- Trained on **windows of consecutive sentences** with global position features kept. At inference the RNN runs over the whole document.
- Class-balanced loss: square-root inverse-frequency weights, optional label smoothing.
- Warm-up and cosine LR, AMP, gradient accumulation, early stopping on mean dev macro-F1.
- **Decoding** (GQLA, TXL): ensemble probability averaging, prior correction (`tau`), and Viterbi transition smoothing (`lambda`). An ensemble subset or decoding setting is adopted only if it beats the simpler choice by a margin (>0.01 for subsets, >0.003 for decoding), to avoid overfitting 5 dev docs.
- Every experiment saves its probabilities to Drive and is **skipped and reloaded** when the notebook is re-run.

## 3. Results (dev, mean macro-F1 over CL and IT)

### Encoder–decoder (encoder only, masked mean pool → BiGRU)
| Exp | Technique | Trainable | CL F1 | IT F1 | **Mean F1** |
|---|---|---|---|---|---|
| exp0 | Frozen encoder + head (probe) | 2.1M | 0.434 | 0.311 | 0.373 |
| exp1 | Full FT | 56.6M | 0.357 | 0.408 | 0.382 |
| exp2 | LoRA r=8 (all attention + FFN linears) | 3.0M | 0.390 | 0.442 | 0.416 |
| **exp3** | **LoRMA⁺** (multiplicative, additive inflation) | 3.1M | 0.379 | 0.469 | **0.424** |
| exp4 | rsLoRA (α/√r) | 3.8M | 0.434 | 0.392 | 0.413 |
| exp5 | IA³ (scales K, V, FFN) | 2.1M | 0.440 | 0.311 | 0.376 |
| exp5b | Prompt tuning (16 soft tokens) | 2.1M | 0.407 | 0.353 | 0.380 |
| exp5c | Partial FT (top 3 layers) | 14.7M | 0.489 | 0.349 | 0.419 |
| exp6 | KD: 9-layer teacher → 4-layer student | 35.7M | 0.370 | 0.425 | 0.398 |
| exp6b | 4-layer student, hard labels only (KD ablation) | 35.7M | 0.414 | 0.396 | 0.405 |

The Phase-1 model has bias-free linears, so BitFit has nothing to tune here. In this run, KD did not beat the same student trained without a teacher.

### GQLA (sentence encoder → document BiLSTM)
| Exp | Technique | Trainable | CL F1 | IT F1 | **Mean F1** | Strict-13 |
|---|---|---|---|---|---|---|
| **E4** | **Houlsby adapters** | 3.5M | 0.469 | 0.568 | **0.518** | 0.439 |
| E3 | LoRMA (`W' = (I + s·B·A)·W`, `B = 0`) | 6.0M | 0.459 | 0.565 | 0.512 | 0.433 |
| E6 | KD: 5-layer student (Phase-1 layers 1,3,5,7,9), teacher = ensemble of E1–E5 | 54.1M | 0.514 | 0.500 | 0.507 | 0.429 |
| E2 | LoRA | 5.6M | 0.464 | 0.526 | 0.495 | 0.419 |
| E5 | BitFit | 2.2M | 0.439 | 0.518 | 0.479 | 0.405 |
| E1 | Full FT | 89.3M | 0.389 | 0.510 | 0.449 | 0.395 |
| E0 / A2 | Frozen features + BiLSTM (3 seeds) | 2.2M | — | — | 0.420–0.431 | — |
| A1 | Frozen, sentence + position (no context) | 0.35M | 0.333 | 0.434 | 0.383 | 0.324 |
| — | TF-IDF + logistic regression | — | 0.349 | 0.409 | 0.379 | 0.334 |
| A0 | Frozen, sentence only | 0.34M | 0.252 | 0.359 | 0.306 | 0.259 |

**Context ablation:** sentence only 0.306 → + position 0.383 → + document BiLSTM 0.420–0.431 → fine-tuned 0.45–0.52.

**Ensemble + decoding:**
- Ensemble of E4 + E3: 0.550
- Adding prior correction `tau=0.25` and Viterbi `lambda=0.3`: **0.554 mean** (CL 0.498, IT 0.609; pooled 0.606; strict-13 0.468)
- These predictions are written to `Phase-3-v2/submission.zip` (10 test docs, 2,197 sentences).

### Transformer-XL + GQA (sentence encoder → 2-layer BiLSTM)
| Exp | Technique | Trainable | CL F1 | IT F1 | **Mean F1** |
|---|---|---|---|---|---|
| A1 | Frozen, sentence + position | 0.49M | 0.283 | 0.221 | **0.252** |
| A2 / E0 | Frozen + BiLSTM | 2.3M | 0.27–0.30 | 0.20–0.22 | 0.247–0.248 |
| lorma | LoRMA⁺ on q,k,v,out | 2.8M | 0.243 | 0.248 | 0.246 |
| prefix_tuning | Prefix tuning through the XL `memory` argument | 2.5M | 0.250 | 0.237 | 0.244 |
| knowledge_distillation | 2-layer student (teacher layers 0 and 7) | 40.2M | 0.224 | 0.251 | 0.237 |
| lora | LoRA on q,k,v,out | 2.9M | 0.239 | 0.228 | 0.233 |
| prompt_tuning | Soft prompt per sentence | 2.3M | 0.267 | 0.200 | 0.233 |
| full_finetune | Full FT | 78.2M | 0.236 | 0.222 | 0.229 |
| A0 | Frozen, sentence only | 0.47M | 0.034 | 0.038 | 0.036 |

Ensemble: frozen BiLSTM + LoRMA gives 0.251. Decoding did not help, so `tau=0` and `lambda=0` were kept.

On this backbone, training loss stays around 2.0, and **no adaptation method beats the frozen features**. Sentence-only features are almost uninformative (0.036). Nearly all the signal comes from position and document context, not from the Transformer-XL sentence representations.

---

## 4. Observations
- **GQLA is the strongest RR backbone.** With ensembling and decoding it reaches 0.554, about double the original isolated-sentence baseline.
- **Document context matters**: position features and the cross-sentence RNN add about 0.12 F1 on GQLA before any fine-tuning.
- **PEFT beats full fine-tuning** on both GQLA and the encoder–decoder, which fits the tiny training set (80 documents): full FT overfits.
- LoRMA is the best or second-best PEFT method on all three backbones.

## 5. How to run
1. Point the config at your Phase-1 checkpoint (`encoder_decoder_best.pt`, `gqla_baseline_large.pt`, or `best_model_transformer_xl_gqa-small.pt`) and `final_bpe_en.json`.
2. Run setup, data and model sections, then the cheap baselines (frozen probes and TF-IDF, a few minutes each). They tell you whether context helps before you spend GPU time.
3. Run E1 → E5, then E6 (KD needs the trained teachers).
4. Run the ensemble and decoding cell, then the write-and-validate cell. It writes `submission.zip` and checks that the prediction count equals the test sentence count.
5. Pick the metric variant (`all13` / `union` / `support`) that matches the official evaluator. `refit=True` (encoder–decoder) retrains on train + dev for the final submission.
