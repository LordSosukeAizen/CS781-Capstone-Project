# Phase 5: Legal Statute Identification (LSI)

**Multi-label** classification: given a long legal document, predict which of **100 statutes** apply.

**Metric:** Macro-F1, the primary metric. Micro-F1 and subset accuracy are also logged. Macro-F1 is reported four ways:
- fixed threshold 0.5
- one **global** threshold tuned on dev (the headline number below)
- per-statute thresholds tuned on dev (optimistic)
- an **honest split-half check**: tune on one half of dev and score on the other

| Notebook | Backbone | Best dev Macro-F1 (global threshold) |
|---|---|---|
| [encoder-decoder-phase-5.ipynb](encoder-decoder-phase-5.ipynb) | Encoder–decoder | 0.292 (frozen, cached BiGRU baseline) |
| [gqla-phase-5.ipynb](gqla-phase-5.ipynb) | GQLA | **0.341** (partial FT / QLoRA) |
| [mamba-phase-5.ipynb](mamba-phase-5.ipynb) | Mamba | 0.101 (AdaLoRA), trained on a 4k-doc subset |
| [transformer-xl-phase-5.ipynb](transformer-xl-phase-5.ipynb) | Transformer-XL + GQA | 0.175 (LoRMA⁺) |

---

## 1. Data
`Exploration-Lab/CS-781-Capstone` (Phase 5 folder): `train.json`, `dev.json`, `test.json`, with fields `id`, `text`, `labels`.

| Split | Docs |
|---|---|
| train | 42,750 (3.70 statutes per doc) |
| dev | 10,181 (3.92 statutes per doc) |
| test | 13,019 |

All 100 statutes have at least 272 training documents (median 679, max 11,730). No statute appears in dev without appearing in train.

## 2. Model: hierarchical, multi-label
```
document → 512-token chunks (the Phase-1 window)
        → Phase-1 backbone (frozen or adapted)  → token states
        → chunk vector  = [mean | max | last]                   (decoder-only models)
                        = [mean | max | <eos> | decoder state]  (encoder–decoder)
        → chunk-level BiGRU / BiLSTM / small BERT-style encoder (cross-chunk context)
        → attention pooling → one sigmoid per statute
```

The decoder-only backbones are causal, so the bidirectionality comes from the chunk-level RNN. In GQLA and Mamba, `token` mode can also add a token-level BiRNN.

Two modes share everything after the backbone:
- **`cached`**: the backbone runs **once** and per-chunk vectors are stored in fp16. Only the head trains, in seconds per epoch. Used for ablations and the KD student.
- **`token`**: the backbone runs every step and is adapted with `peft=dict(method=...)`.

Rare-statute-aware losses (`bce_pw`, `asl`), a force-at-least-one-statute rule, and threshold tuning. Self-test cells check that every PEFT method starts as the identity of the Phase-1 backbone, and that the fast attention or scan paths match the Phase-1 maths.

## 3. The six experiments (brief §11.2)
| # | Name | Technique |
|---|---|---|
| E1 | `e1_lorma_plus` | **LoRMA⁺** `(I + α/r·BA)·W₀` (compulsory) on the top blocks |
| E2 | `e2_adalora` | **AdaLoRA**: SVD-form update with importance-based rank pruning, rank 12 → 6 |
| E3 | `e3_qlora` | **QLoRA**: whole backbone stored in 4-bit NF4, LoRA on top |
| E4 | `e4_adapter` | **Houlsby bottleneck adapters** (after attention and after the FFN) |
| E5 | `e5_partial_ft` | **Partial FT**: top 3 blocks + final norm |
| E6 | `e6_kd_student` | **KD**: truncated backbone (first 4–5 blocks, frozen) + small BiGRU; teacher = ensemble of E1–E5 (T=2, λ=0.5) |

Optional extras: LoRA, LoRMA-π, prefix tuning, prompt tuning, full FT, and the student without a teacher.

## 4. Results (dev)

### GQLA
| Exp | Macro-F1 | Micro-F1 | Honest split-half | Trainable | Deployed | Train time |
|---|---|---|---|---|---|---|
| **E5 partial FT** | **0.341** | 0.458 | 0.339 | 23.2M | 89.2M / 357 MB | 259 min |
| E3 QLoRA | 0.341 | 0.456 | 0.339 | 3.5M | 90.6M / **119 MB** | 329 min |
| E2 AdaLoRA | 0.340 | 0.454 | 0.339 | 3.1M | 90.3M / 361 MB | 363 min |
| E4 adapters | 0.336 | 0.448 | 0.335 | 2.7M | 89.8M / 359 MB | 266 min |
| E1 LoRMA⁺ | 0.330 | 0.446 | 0.328 | 2.5M | 89.6M / 359 MB | 303 min |
| E6 KD student | 0.294 | 0.415 | 0.293 | **0.75M** | 52.7M / 211 MB | **5.5 min** |

The five PEFT and partial-FT methods land within 0.011 of each other. QLoRA matches the best score at a third of the storage. The KD student keeps 86% of the best score with 59% of the parameters and inference about 300× faster (1.3 s vs ~430 s on dev).

### Encoder–decoder
| Exp | Macro-F1 | Micro-F1 | Honest | Trainable | Train time |
|---|---|---|---|---|---|
| **Cached BiGRU (frozen encoder)** | **0.292** | — | 0.291 | — | — |
| E6 KD student (4 encoder layers) | 0.265 | 0.383 | 0.265 | 0.75M | 5.9 min |
| E5 partial FT | 0.246 | 0.372 | 0.246 | 14.7M | 106 min |
| E1 LoRMA⁺ | 0.228 | 0.351 | 0.226 | 2.2M | 115 min |
| E2 AdaLoRA | 0.228 | 0.362 | 0.227 | 2.7M | 161 min |
| E4 adapters | 0.187 | 0.325 | 0.186 | 2.7M | 114 min |
| E3 QLoRA | 0.181 | 0.303 | 0.181 | 2.9M | 167 min |

The frozen-feature baseline beats every adapted model. With these placeholder hyper-parameters (4 epochs), token-mode adaptation hurt rather than helped. The encoder-only chunk format is 511 tokens + `<eos>`, matching the Phase-1 encoder inputs.

### Mamba
| Exp | Macro-F1 | Micro-F1 | Note |
|---|---|---|---|
| E2 AdaLoRA | 0.101 | 0.167 | |
| E4 adapters | 0.079 | 0.112 | |
| E1 LoRMA⁺ | 0.079 | 0.094 | |
| E3 QLoRA | 0.077 | 0.111 | |
| E5 partial FT | 0.116 after 3 epochs | 0.201 | did not finish |
| E6 KD | — | — | not run |

- Trained on **4,000 of 42,750** documents to fit the time budget.
- The fused `mamba_ssm` scan kernel was **OFF** in this run (it fell back to the pure-PyTorch scan, in bf16).
- Scores near 0.08 with a tuned threshold of about 0.1 mean the model mostly predicts label frequencies.

### Transformer-XL + GQA
| Exp | Macro-F1 | Micro-F1 | Note |
|---|---|---|---|
| E1 LoRMA⁺ | **0.175** | 0.305 | 726 min to train |
| Cached BiGRU (frozen) | 0.117 | 0.226 | |
| E3 QLoRA | 0.080 | 0.113 | 780 min |
| E2 AdaLoRA, E4 adapters | — | — | stopped after 1–2 epochs (~3 h per epoch) |
| E5, E6 | — | — | not run |

Transformer-XL specifics:
- Chunks are `<s>` + 510 tokens + `</s>`.
- A memory-lean attention computes `(q+u)·k` inside SDPA and passes `shift((q+v)·r)` as the additive mask. It is self-tested against the Phase-1 attention.
- Prefix tuning uses learned XL memory.
- The KD student is the first 4 of 8 blocks.
- Chunks are encoded independently (no segment recurrence).

> The Mamba and TXL logs print `AdaLoRA ranks now: min 0 / mean 0.0 / max 0`. Check whether the rank allocator pruned every rank before treating E2 on those backbones as a real AdaLoRA result.

---

## 5. Observations
- **GQLA is the best LSI backbone** (0.34). It is also the only one where adaptation clearly beats the frozen baseline.
- **Cheap KD students are attractive**: the cached-mode student trains in about 5 minutes and stays within 0.03–0.05 of the teachers.
- No statute is rare (at least 272 training docs each), so the "F1 by train frequency" analysis falls into a single bucket (101+ docs).

## 6. How to run
1. Run top to bottom. You need the Phase-1 checkpoint, `final_bpe_en.json`, and an `HF_TOKEN` Colab secret. Edit `GQLA_CFG` / `MAMBA_CFG` if your Phase-1 hyper-parameters differ.
2. Run the self-test cells (fast attention or scan vs Phase-1 maths, PEFT identity at init).
3. Run one experiment first and read its `s/batch → ~N min per epoch` line before queueing the rest. Token-mode runs take hours on a T4.
4. Each run saves `result_<name>.pkl` to Drive and is reused on re-run (`force=True` to retrain).
5. The summary cell writes `submission_<name>.zip` for each experiment (13,019 predictions).

All hyper-parameters in the RUN cell are placeholders (4 epochs, ranks, learning rates, temperature), chosen only so the notebook runs end to end.
