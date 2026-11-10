# Phase 1: Legal Language-Model Pretraining

Phase 1 pretrains four language-model architectures **from scratch** on Indian legal text. Every later phase (2–7) starts from one of these checkpoints. No external pretrained weights are used anywhere in the project.

| Notebook | Architecture | Params | Checkpoint used downstream |
|---|---|---|---|
| [encoder-decoder-phase-1.ipynb](encoder-decoder-phase-1.ipynb) | T5-style encoder–decoder (`LegalSeq2SeqLM`) | 101.7M | `encoder_decoder_best.pt` |
| [gqla-phase-1.ipynb](gqla-phase-1.ipynb) | Decoder-only, Group-Query **Latent** Attention (GQLA) | 103.9M | `best_model.pt` / `gqla_baseline_large.pt` |
| [mamba-phase-1.ipynb](mamba-phase-1.ipynb) | Decoder-only, Mamba selective state-space model | 86.2M | `mamba_large.pt` |
| [transformer-xl-gqa-phase-1.ipynb](transformer-xl-gqa-phase-1.ipynb) | Decoder-only, Transformer-XL + Grouped-Query Attention | 76.0M ("small") | `best_model_transformer_xl_gqa-small.pt` |

---

## 1. Data

Source: the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`, folder `Phase_1_Pretraining/`.

| File | Documents scanned |
|---|---|
| `Train/train_acts.json` | 18,826 |
| `Train/train_high_courts.json` | 2,536,738 |
| `Train/train_supreme_court.json` | 55,549 |
| `Train/train_tribunals.json` | 869,132 |
| `Test/test.json` | 1,000 test prompts |

Each JSON array is streamed with `ijson` and **reservoir-sampled** per source. The per-source caps differ slightly between notebooks:

| Notebook | Acts / HC / SC / Tribunals cap | Sampled | After cleaning | Train / Val |
|---|---|---|---|---|
| Encoder–decoder | all / 300k / 50k / 300k | 668,826 | 613,517 | 582,842 / 30,675 |
| GQLA | all / 300k / 50k / 250k | 618,826 | 570,115 | 541,610 / 28,505 |
| Transformer-XL | all / 300k / 50k / 200k | 568,826 | 526,311 | 499,996 / 26,315 |
| Mamba | reuses the cached token files | — | — | 499,996 / 26,315 |

### Cleaning (same code in all four notebooks)
Cleaning runs **before** the train/validation split, so duplicates cannot leak from train into validation.
1. **Normalisation**: curly quotes and dashes become ASCII. NFKC fixes nbsp, ligatures and full-width forms. Zero-width and control characters are removed. Whitespace runs collapse to one space, and runs such as `.......` are shortened to 3.
2. **Language filter**: documents that are not mostly Latin script or not recognisably English are dropped. Stray Hindi/Urdu fragments inside English documents are stripped.
3. **Deduplication**: exact duplicates (after normalisation) and near-duplicates (MinHash-LSH, ~0.8 Jaccard) are removed.

`normalize_text` is also applied to test prompts at inference, so train and test text match.

### Tokenizer
A custom `BPETokenizer` (word-level pre-tokenisation with `WORD_RE = \w+|[^\w\s]`, then BPE merges) with a **32,768** vocabulary, saved as `final_bpe_en.json`. It is trained on training documents only. The same tokenizer file is reused by every later phase, so all four models share token ids.

---

## 2. Models

### Encoder–decoder (`LegalSeq2SeqLM`)
- 9 encoder + 9 decoder layers, `d_model=512`, 16 heads, `d_ff=2048`, dropout 0.1
- RMSNorm, T5-style bucketed relative position bias, gated-GELU FFN, tied embeddings
- Objective: **prompt → continuation**. The encoder reads 512 BPE tokens and the decoder predicts the next 128. Decoder input is `[BOS] + target[:-1]`.
- Training: batch 2 × grad-accum 8, AdamW lr 1e-4, wd 0.01, 200 warm-up steps, clip 1.0, AMP, validation and checkpoints every 200 steps (resumable).

### GQLA
- 10 layers, `d_model=512`, 64 query heads, 32 KV groups, head dim 64, **latent KV dim 96**, decoupled RoPE (dim 32)
- 512-token chunks, batch 4 × grad-accum 16, AdamW lr 1e-4, wd 0.01, 3 epochs planned
- Also tries two extra decoding strategies at test time: **speculative decoding** (Leviathan et al., 2023) and **FSD-d** (Frustratingly Simple Decoding, discrete anti-LM).

### Mamba
- Implemented from the paper (Gu & Dao, 2023) in plain PyTorch, with no custom CUDA build.
- 20 Mamba blocks (about the depth of 10 Transformer layers), `d_model=512`, `d_state=32`, `d_conv=16`, `expand=3`, dropout 0.1
- AdamW lr 3e-4 (betas 0.9/0.95), 5% warm-up and cosine decay to 10%
- Same data, tokenizer, split, 512-token chunking and test protocol as GQLA, so the comparison is fair.

### Transformer-XL + GQA
- Segment recurrence with relative sinusoidal positions (`u`/`v` biases) and grouped-query attention. The sequential loader keeps each batch row a continuous token stream, so the memory from segment *t* belongs to the same stream at segment *t+1*.
- Segment length 128, memory length 128, batch 6 × grad-accum 32, AdamW lr 1e-4.
- Two runs were trained:
  - **small**: 8 post-LN blocks, `d=768`, 32 query / 8 KV heads, **76.0M params**. Later phases use this one (Phases 2–5 and 7).
  - **large**: 32 layers, `d=512`, 64 query / 16 KV heads, 107.1M params. **It diverged** (see results).

> The configuration cell in the notebook currently shows the *large* settings, because they were the last ones run. The small checkpoint's settings are the ones listed above, as confirmed by the later phases that load it.

---

## 3. Results

Test protocol (GQLA, Mamba and Transformer-XL): 512-token prompt → 128-token greedy continuation, scored against the true continuation.

| Model | Validation | Test BERTScore | Test ROUGE-L | Test BLEU-4 |
|---|---|---|---|---|
| Encoder–decoder | token loss **4.872** (ppl 130.6) at step 11,200 | — | — | — |
| GQLA | train loss ≈ 3.98 (ppl ≈ 54) at step 11,000 | 0.732 | 0.002 | 0.000 |
| **Mamba** | train loss ≈ 3.9 (ppl ≈ 49) at step ~1,450 | **0.805** | **0.176** | **0.0185** |
| Transformer-XL (small) | val loss 3.487 (ppl 32.7) after epoch 5; checkpoint best val loss 3.077 | 0.787 | 0.135 | 0.0094 |
| Transformer-XL (large) | val loss 6.981 (ppl **1,076**) after 5 epochs | — | — | — |

- The encoder–decoder notebook writes the official `prediction.json` and `submission.zip` for the 1,000 test prompts but does not score them locally.
- GQLA's speculative and FSD-d decoding variants all score about the same (BERTScore 0.732–0.733, ROUGE-L ≈ 0.002).

### Why GQLA's test scores are near zero
The Mamba notebook documents a **double-shift bug in the GQLA training loss**. `GQLADataset` already returns `labels = chunk[1:]`, and the loss then shifts again (`logits[:, :-1]` vs `labels[:, 1:]`). As a result, GQLA learned to predict the token **two** positions ahead. Its continuations are therefore misaligned, which explains ROUGE-L ≈ 0.002 despite a reasonable loss. The Mamba notebook computes the loss directly on `(logits, labels)`.

Downstream phases use GQLA as a feature extractor and fine-tune it, so the bug matters less there. GQLA is still among the strongest backbones in Phases 2–5 and 7.

### Perplexities are not directly comparable
The encoder–decoder predicts a 128-token continuation from a 512-token encoder context. GQLA's loss is for a two-ahead target. Transformer-XL uses 128-token segments with memory. Only the shared test protocol (BERTScore / ROUGE-L / BLEU-4) is comparable across models.

---

## 4. How to run

Notebooks were written for **Google Colab (T4 GPU) with Google Drive mounted**.

1. Put a Hugging Face token in a Colab secret named `HF_TOKEN`.
2. Edit the Drive paths in the first config cell (`DRIVE_ROOT`, `cache_dir`, and so on).
3. Run top to bottom. Data preparation is cached. The Mamba notebook skips preparation when it finds the cached token files ("CACHED").
4. Training is resumable from the latest checkpoint after a Colab disconnect.

Dependencies: `torch`, `ijson`, `huggingface_hub`, `datasets`, `pyarrow`, `bert-score`, `rouge-score`, `sacrebleu`. MinHash-LSH deduplication and language filtering are implemented inside the notebooks.

## 5. Outputs (on Drive)
- `final_bpe_en.json`: tokenizer shared by all phases
- Checkpoints: `encoder_decoder_best.pt`, `best_model.pt` (GQLA), `mamba_large.pt`, `best_model_transformer_xl_gqa-small.pt`
- `prediction.json` / `truth.json` and `submission.zip` with `prediction.json` at the zip root
