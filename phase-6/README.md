# Phase 6: Abstractive Legal Summarisation

Each Phase-1 model is turned into an **abstractive summariser** for long court judgments, with no new pretraining.

**Metrics:** BERTScore (computed with `roberta-large`) and ROUGE-1 / ROUGE-2 / ROUGE-L. Results below are on a **351-document validation split** held out from `train.json`. The test set has 100 documents.

| Notebook | Backbone | How it reads a long document | Best val ROUGE-1 / ROUGE-L / BERTScore |
|---|---|---|---|
| [encoder-decoder-phase-6.ipynb](encoder-decoder-phase-6.ipynb) | Encoder–decoder | **Fusion-in-Decoder** over K chunks | **0.437 / 0.215 / 0.821** (full FT) |
| [gqla-phase-6.ipynb](gqla-phase-6.ipynb) | GQLA | Compressed memory as soft-token prefix | 0.398 / 0.208 / **0.835** (KD) |
| [mamba-phase-6.ipynb](mamba-phase-6.ipynb) | Mamba | Compressed memory as prefix, recurrent decoding | 0.424 / 0.211 / 0.823 (LoRMA⁺) |
| [transformer-xl-phase-6.ipynb](transformer-xl-phase-6.ipynb) | Transformer-XL + GQA | Compressed memory as prefix | 0.373 / 0.215 / — (KD) |

---

## 1. Architectures

### Encoder–decoder: Fusion-in-Decoder (FiD)
```
document → K chunks (511 BPE tokens + </s>, spread uniformly over the judgment)
        → ENCODER on each chunk independently   (the 512-token window it was pretrained on)
        → concatenate all chunk states [B, K·512, d] (+ key mask)
        → DECODER cross-attends to ALL chunks at once → summary
```
No new weights. Decoder length can exceed 512 because the T5 relative-position buckets are log-spaced. Incremental beam search uses a self-attention KV cache and computes cross-attention K/V **once per document**, shared by all beams.

### Decoder-only (GQLA, Mamba, Transformer-XL): memory bridge
```
document → 512-token chunks (K spread over the judgment)
        → frozen Phase-1 model, run once → hidden states per chunk
        → m segment-means per chunk      → K·m memory slots (cached fp16 on Drive)
        → MemoryBridge (trainable, ~3M): chunk/slot positions + 1 bidirectional
                                         Transformer layer + projection to embedding space
        → adapted model reads ONE causal sequence:
          [ prompt? | pad … | mem_1 … mem_N | <s> | y_1 … y_L | </s> ]
                                                 └── loss only here (label smoothing 0.1)
```
- **GQLA and TXL** must fit memory and summary into 512 positions, which leaves about 380 summary tokens with 128 memory slots.
- **Mamba** has no 512 window. Its recurrent step uses constant memory per token, so summaries can be as long as the references (capped at 768). `mem_mode="text"` feeds 2,048 raw tokens instead, an ablation only a recurrent model can run cheaply.
- `mem_mode="lead"` (GQLA, TXL) is the plain prefix-LM baseline that feeds the first document tokens as text.

### Fixes relative to Phase 5
- **GQLA:** fused SDPA during training as well (with dropout passed to the kernel), which stops full FT from running out of memory. The KV cache keeps un-rotated RoPE keys (the Phase-1 cache rotated them twice). PEFT parameters now actually receive gradients.
- **Mamba:** a left-padding mask keeps the SSM state at zero through padding. `mixer_step` updates the conv window and SSM state one token at a time for recurrent decoding.
- **TXL:** a new attention forward with a grouped KV cache (positions are recomputed from key length). Prefix tuning goes through the XL `memory` slot.

### Methods
`full` · `partial` · `lora` · **`lorma` (LoRMA⁺, compulsory)** · `qlora` (NF4 base + LoRA) · `houlsby` · `pfeiffer` · `prefix` · `prompt` · `frozen` (bridge only) · **`kd`** (student initialised from the fine-tuned teacher, trained with CE + temperature-scaled KL on the teacher's logits).

KD students: encoder–decoder 6 encoder / 4 decoder layers; GQLA 5 of 10 layers; Mamba 10 blocks; TXL 4 of 8 blocks `[0, 2, 5, 7]`.

## 2. Results (val, 351 docs)

### Encoder–decoder (FiD)
| Exp | Method | Trainable | BERTScore | R-1 | R-2 | R-L | Avg words |
|---|---|---|---|---|---|---|---|
| **E1** | **Full FT** | 101.7M | 0.821 | **0.437** | **0.181** | **0.215** | 400 |
| E2 | LoRA r=16 (162 linears) | 3.98M | **0.821** | 0.438 | 0.181 | 0.214 | 385 |
| E3 | LoRMA⁺ r=16 | 4.42M | 0.818 | 0.427 | 0.176 | 0.212 | 394 |
| E6 | KD (6-enc / 4-dec student) | 62.9M | 0.817 | 0.421 | 0.176 | 0.213 | 340 |
| E4 | Houlsby adapters (b=64) | 2.38M | 0.807 | 0.423 | 0.166 | 0.212 | 467 |
| E5 | Prefix tuning (16 vectors, enc + dec) | 0.15M | 0.803 | 0.392 | 0.150 | 0.202 | 366 |

LoRA ties full FT while training 4% of the weights.

### GQLA
| Exp | Method | Trainable | BERTScore | R-1 | R-2 | R-L | Avg words |
|---|---|---|---|---|---|---|---|
| **G6** | **KD (5-layer student)** | 72.2M | **0.835** | **0.398** | **0.162** | **0.208** | 288 |
| G3 | LoRMA⁺ r=16 | 7.25M | 0.830 | 0.377 | 0.142 | 0.201 | 280 |
| G2 | LoRA r=16 | 6.84M | 0.832 | 0.372 | 0.143 | 0.200 | 271 |
| G1 | Full FT | — | 0.831 | 0.373 | 0.141 | 0.199 | 270 |
| G4 | Houlsby adapters | 4.75M | 0.831 | 0.355 | 0.138 | 0.196 | 247 |
| G5 | Prefix tuning | 3.51M | 0.808 | 0.342 | 0.117 | 0.198 | 317 |

The KD student beats its own teacher (G1) on every metric.

### Mamba
| Exp | Method | Trainable | BERTScore | R-1 | R-2 | R-L | Avg words |
|---|---|---|---|---|---|---|---|
| **M3** | **LoRMA⁺ r=16** | 5.78M | 0.823 | **0.424** | **0.155** | **0.211** | 492 |
| M2 | LoRA r=16 | 5.75M | **0.825** | 0.410 | 0.152 | 0.208 | 393 |
| M1 | Full FT | 89.7M | 0.825 | 0.412 | 0.150 | 0.207 | 425 |
| M5 | Prompt tuning (20 soft tokens) | 3.44M | — | 0.374 | 0.119 | 0.195 | 660 |
| M4 | Adapters after every mixer | 4.75M | did not finish (val-subset R-1 0.416 at epoch 5) | | | | |
| M6 | KD (10-block student) | 63.3M | did not finish (3 epochs, val-subset R-1 0.413) | | | | |

Mamba's summaries are the longest (390–490 words), closest to the 825-word reference average.

### Transformer-XL + GQA
| Exp | Method | Trainable | BERTScore | R-1 | R-2 | R-L |
|---|---|---|---|---|---|---|
| **T6** | **KD (4 of 8 blocks)** | 31.5M | — | **0.373** | **0.125** | **0.215** |
| T1 | Full FT | — | 0.724 | 0.271 | 0.055 | 0.202 |
| T5 | Prefix (16 learned XL memory slots) | 3.69M | — | 0.265 | 0.051 | 0.182 |
| T3 | LoRMA⁺ r=16 | 7.36M | — | 0.261 | 0.047 | 0.177 |
| T4 | Houlsby adapters | 7.66M | — | 0.259 | 0.046 | 0.177 |
| T2 | LoRA | 7.75M | stopped after 1 epoch | | | |

> **Wrong checkpoint.** This notebook loads `transformer_xl_gqa-large.pt` (107.1M params, `best_val_loss 6.98`). That is the **diverged** Phase-1 run (val ppl ≈ 1,076), not the `…-small.pt` checkpoint (76.0M, best val loss 3.08) that Phases 2–5 and 7 use. Training losses stuck at ≈ 7.3 and R-2 ≈ 0.05 follow from this. **Re-point `CHECKPOINT_PATH` to `best_model_transformer_xl_gqa-small.pt` and re-run** before you compare TXL with the other backbones.

---

## 3. Observations
- The **encoder–decoder (FiD) gives the best ROUGE**. Cross-attention over all chunks is a natural fit for long-document summarisation.
- **GQLA's KD student gives the best BERTScore** (0.835) and beats its teacher. **Mamba with LoRMA⁺** is a close second on ROUGE.
- Every model writes summaries that are too short (270–490 words vs an 825-word reference average). Length control is an obvious next step.
- PEFT (LoRA / LoRMA⁺) matches full FT on all three working backbones.

## 4. How to run
1. Run top to bottom. The self-test cell (1b / 2b) checks the new attention or FiD runners against the Phase-1 forward, cached decoding against full decoding, and left padding. The config cell checks that every PEFT module is the identity at init.
2. Run the smoke test. The first real run extracts and caches the memory features for train, val and test (a one-off of about 10–20 min on a T4).
3. Run the experiments one cell at a time. Each saves its checkpoint, val predictions and `submission_<name>/submission.zip` to Drive.

Dependencies: `rouge-score`, `bert-score` (downloads `roberta-large` for scoring only).
