# Phase 7: Prior Case Retrieval (PCR)

Given a query case and its split's candidate pool, return the **7 most relevant prior cases**.

**Metric:** **micro-F1@7**. P@7, R@7, R@50, R@100 and MRR are also logged. All numbers below are on dev.

| Notebook | Backbone | Best dense F1@7 | Best hybrid F1@7 |
|---|---|---|---|
| [encoder-decoder-phase-7.ipynb](encoder-decoder-phase-7.ipynb) | Encoder–decoder (encoder only) | 0.310 (LoRMA⁺) | 0.339 (full FT) |
| [gqla-phase-7.ipynb](gqla-phase-7.ipynb) | GQLA | **0.325** (partial FT) | **0.361** (partial FT) |
| [transformer-xl-phase-7.ipynb](transformer-xl-phase-7.ipynb) | Transformer-XL + GQA | 0.030 (full FT) | 0.237 (KD) |

There is no Mamba notebook for this phase.

---

## 1. Model: bi-encoder retriever
```
case → 512-token windows (8 spread over the judgment at test time; 4 random ones per case in training)
     → Phase-1 backbone (adapted) → final hidden states of each window
     → attention pooling over tokens (zero-init = mean) → projection (identity init) → unit window vectors
     → attention aggregation over windows (zero-init = mean) → one unit vector per case
score(query, candidate) = cosine     (or "maxsim": window-level late interaction)
```
- Each pool is encoded once, and ranking is a matrix product.
- Because of the zero and identity initialisations, the **untrained retriever is exactly mean-pooled Phase-1 states**. The zero-shot run (`D0` / `Q0` / `X0`) measures how well Phase-1 representations transfer, and every method starts from that point.
- The encoder–decoder drops its decoder and uses chunks of 511 tokens + `</s>`. Transformer-XL uses `<s>` + 510 + `</s>`, with windows encoded independently (no segment recurrence).

## 2. Training
- **InfoNCE** over the batch. Each query has 1 relevant case and **3 BM25 hard negatives** (top non-relevant cases from its pool), and every other case in the batch is an in-batch negative. Cases relevant to the query are never used as negatives.
- 817 training queries, batch of 8 queries, 3 epochs.
- A 40-case batch means about 160 window forwards, so the notebooks use **GradCache** (Gao et al., 2021): the exact full-batch gradient with the memory of one sub-batch. A self-test verifies it against ordinary backprop.
- **KD**: a half-depth student initialised from the fine-tuned teacher. It learns from the teacher's score distribution over each batch (listwise KL), from the teacher's case vectors (cosine alignment), and from the teacher's hard negatives.
- **BM25** over the project's own BPE ids (unigrams + hashed bigrams) serves three purposes: a lexical reference, the source of hard negatives, and an optional **hybrid** that combines z-normalised dense and BM25 scores, with the weight tuned on dev. Hybrid results are written as separate `_hybrid` submissions. Every system still ranks with the fine-tuned Phase-1 model.

**Techniques:** `full` · `partial` (top 3 blocks + final norm) · **`lorma` (LoRMA⁺, compulsory)** · `ia3` · `prefix` · `prompt` · `qlora` · **`kd`**, plus `lora` as an extra.

## 3. Results (dev)

### Encoder–decoder
| Exp | Method | Trainable | F1@7 | R@100 | MRR | Hybrid F1@7 (w_dense) |
|---|---|---|---|---|---|---|
| D0 | Zero-shot (mean-pooled Phase-1 encoder) | 0.26M | 0.209 | 0.659 | 0.536 | 0.259 |
| D1 | Full FT | 54.8M | 0.309 | 0.750 | 0.710 | **0.339** (0.7) |
| **D2** | **LoRMA⁺ r=16** (63 linears) | 2.18M | **0.310** | 0.765 | 0.706 | 0.335 (0.8) |
| D4 | Prefix tuning (16 vectors) | 0.34M | 0.300 | 0.761 | 0.687 | 0.326 (0.8) |
| D3 | IA³ (27 vectors) | 0.29M | 0.299 | 0.756 | 0.693 | 0.329 (0.7) |
| D5 | Partial FT | 12.9M | 0.299 | 0.752 | 0.676 | 0.330 (0.7) |
| D6 | KD (half-depth student) | 38.0M | 0.292 | **0.777** | 0.669 | 0.316 (0.8) |

### GQLA
| Exp | Method | Trainable | F1@7 | R@100 | MRR | Hybrid F1@7 (w_dense) |
|---|---|---|---|---|---|---|
| — | BM25 reference | — | 0.203 | 0.581 | 0.590 | — |
| Q0 | Zero-shot | 0.26M | 0.276 | 0.778 | 0.650 | 0.305 |
| **Q5** | **Partial FT** (top 3 blocks) | 21.4M | **0.325** | 0.806 | 0.722 | **0.361** (0.7) |
| Q2 | LoRMA⁺ r=16 (100 linears) | 4.08M | 0.324 | 0.800 | 0.710 | 0.350 (0.8) |
| Q1 | Full FT | 87.4M | 0.322 | **0.811** | 0.722 | 0.360 (0.7) |
| Q3 | IA³ (40 vectors) | 0.33M | 0.322 | 0.798 | 0.707 | 0.345 (0.9) |
| Q4 | Prefix tuning | 0.35M | 0.317 | 0.809 | 0.711 | 0.335 (0.9) |
| — | QLoRA, KD | — | not run | | | |

IA³ matches full FT with 0.4% of its trainable parameters. Zero-shot GQLA already beats BM25 by 0.07.

### Transformer-XL + GQA
| Exp | Method | Trainable | F1@7 | R@100 | MRR | Hybrid F1@7 (w_dense) |
|---|---|---|---|---|---|---|
| — | BM25 reference | — | **0.202** | 0.582 | 0.586 | — |
| X0 | Zero-shot | 0.59M | 0.011 | 0.168 | 0.049 | 0.203 |
| X1 | Full FT | 76.6M | 0.030 | 0.223 | 0.120 | 0.236 (0.5) |
| X6 | KD | 51.2M | 0.026 | 0.282 | 0.090 | **0.237** (0.7) |
| X5 | Partial FT | 19.6M | 0.023 | 0.177 | 0.086 | 0.205 (0.3) |
| X4 | Prefix (16 XL memory slots) | 0.69M | 0.021 | 0.220 | 0.066 | 0.203 (0.1) |
| X3 | IA³ | 0.62M | 0.013 | 0.145 | 0.051 | 0.202 (0.0) |
| X2 | LoRMA⁺ | 2.07M | 0.009 | 0.104 | 0.035 | 0.202 (0.0) |

Mean-pooled Transformer-XL states are **near random** for retrieval (zero-shot F1@7 0.011, far below BM25's 0.202). Training loss for most PEFT runs stays flat at about 3.44. Any hybrid gain comes from BM25: for most runs the tuned dense weight is 0.0–0.3. Worth investigating, starting with the per-window feature scale and pooling. This notebook uses the healthy `…-small.pt` checkpoint, so the checkpoint is not the cause here.

---

## 4. Observations
- **GQLA is the best retriever** (dense 0.325, hybrid 0.361). The encoder–decoder is close behind (0.310 / 0.339).
- On both working backbones, cheap PEFT (LoRMA⁺, IA³, prefix) lands within 0.01 of full FT.
- **The hybrid consistently adds 0.025–0.04 F1@7**, so dense and lexical signals are complementary.

## 5. How to run
1. Run top to bottom. The checks cell verifies the encoder runner or attention patch against Phase 1, PEFT identity at init, fp16 against fp32 (TXL), and GradCache against direct gradients.
2. The run cell starts with BM25, the zero-shot model and a smoke test, then runs the experiments one cell at a time.
3. Each experiment writes `submission_<name>/submission.zip` and `submission_<name>_hybrid/submission.zip` (237 test queries × 7 ids).
