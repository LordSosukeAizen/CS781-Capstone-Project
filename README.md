# CS781 Capstone: Legal Foundation Models, Pretrained from Scratch and Adapted to Seven Tasks

This project pretrains **four language-model architectures from scratch** on Indian legal text (Phase 1). It then adapts each of them to **six downstream legal NLP tasks** (Phases 2–7) with full fine-tuning, parameter-efficient fine-tuning (PEFT) and knowledge distillation.

Every downstream experiment starts from a Phase-1 checkpoint. No external pretrained model, pretrained weights or LLM API is used. The one exception is `roberta-large`, which BERTScore uses for scoring only.

## Architectures

| Name | Type | Size | Key ideas |
|---|---|---|---|
| **Encoder–decoder** (`LegalSeq2SeqLM`) | T5-style seq2seq | 101.7M (9 + 9 layers, d=512) | RMSNorm, bucketed relative position bias, gated-GELU FFN, tied embeddings |
| **GQLA** | Decoder-only | 103.9M (10 layers, d=512) | Group-Query **Latent** Attention: 64 query heads, 32 KV groups, latent KV dim 96, decoupled RoPE |
| **Mamba** | Decoder-only, selective SSM | 86.2M (20 blocks, d=512) | Selective scan (S6), d_state 32, d_conv 16, expand 3. No attention and no positional encoding |
| **Transformer-XL + GQA** | Decoder-only, recurrent memory | 76.0M (8 blocks, d=768) | Segment recurrence, relative sinusoidal attention with `u`/`v` biases, grouped-query attention (32q / 8kv) |

All four share one custom **32,768-token BPE tokenizer** (`final_bpe_en.json`), so token ids are identical across models.

## Phases and headline results

| Phase | Task | Metric (dev / val) | Enc–Dec | GQLA | Mamba | TXL |
|---|---|---|---|---|---|---|
| [1](phase-1/README.md) | LM pretraining | test BERTScore / ROUGE-L | val ppl 130.6 | 0.732 / 0.002 | **0.805 / 0.176** | 0.787 / 0.135 |
| [2](phase-2/README.md) | Legal NER (12 types) | strict Macro-F1 | 0.231 | 0.371 | **0.494** | 0.253 |
| [3](phase-3/README.md) | Rhetorical roles (13) | mean Macro-F1 (CL, IT) | 0.424 | **0.518** (ens. 0.554) | — | 0.252 |
| [4](phase-4/README.md) | Judgment prediction (binary) | Macro-F1 @0.5 | 0.552 | **0.736** | 0.584 | 0.333 |
| [5](phase-5/README.md) | Statute identification (100, multi-label) | Macro-F1 | 0.292 | **0.341** | 0.101 | 0.175 |
| [6](phase-6/README.md) | Abstractive summarisation | ROUGE-1 / ROUGE-L / BERTScore | **0.437 / 0.215** / 0.821 | 0.398 / 0.208 / **0.835** | 0.424 / 0.211 / 0.823 | 0.373 / 0.215 / — |
| [7](phase-7/README.md) | Prior case retrieval | micro-F1@7 dense (hybrid) | 0.310 (0.339) | **0.325 (0.361)** | — | 0.030 (0.237) |

Numbers are the best single experiment per backbone on the local dev or validation split, as recorded in the notebook outputs. Official Codabench test scores are not stored in this repository. "—" means no notebook or no metric for that cell. See each phase README for every experiment, and for which runs are incomplete.

## Repository layout

```
project/
├── README.md                 ← this file
├── phase-1/  Pretraining              encoder-decoder · gqla · mamba · transformer-xl-gqa
├── phase-2/  Legal NER                encoder-decoder · gqla · mamba · transformer-xl-gqa
├── phase-3/  Rhetorical roles         encoder-decoder · gqla · transformer-xl-gqa
├── phase-4/  Judgment prediction      encoder-decoder · gqla · mamba · transformer-xl
├── phase-5/  Statute identification   encoder-decoder · gqla · mamba · transformer-xl
├── phase-6/  Summarisation            encoder-decoder · gqla · mamba · transformer-xl
└── phase-7/  Prior case retrieval     encoder-decoder · gqla · transformer-xl
```

Each folder holds one Jupyter notebook per backbone (`<arch>-phase-<n>.ipynb`) and a `README.md`.

## Shared methodology (Phases 2–7)

- **Starting point:** a *fresh copy* of the Phase-1 checkpoint for every experiment. Architecture classes are copied unchanged into each notebook so `load_state_dict` matches strictly.
- **At least 6 methodologically different experiments per backbone**, including the two compulsory ones:
  - **LoRMA⁺** (Bihany, Patel & Modi, ACL Findings 2025): a multiplicative low-rank update `W = (I + (α/r)·B·A)·W₀`, computed as `h = W₀x; y = h + (α/r)·B(A·h)` with `B = 0`, so every adapted layer equals Phase 1 at init.
  - **Knowledge distillation**: a frozen fine-tuned teacher (or teacher ensemble) and a smaller student **initialised from Phase-1 layers**, never from random weights.
- **Other techniques used:** full FT, partial and layer-wise FT, gradual unfreezing, LoRA, rsLoRA, AdaLoRA, QLoRA (NF4), Houlsby and Pfeiffer adapters, IA³, BitFit, prefix tuning (through the Transformer-XL `memory` slot for TXL) and prompt tuning.
- **Self-tests** in most notebooks check that fast attention or scan paths reproduce the Phase-1 maths, and that every PEFT module is the identity at initialisation.
- **Causal backbones on bidirectional tasks:** GQLA, Mamba and TXL get bidirectionality from a head on top (BiGRU, BiLSTM or a small Transformer), because their states only see left context.
- **Long documents:** chunked into 512-token windows, the Phase-1 context length, then aggregated hierarchically. The exception is Phase 6, which uses FiD or memory compression.
- **Submissions:** every experiment writes a validated `submission.zip` with `prediction.json` at the zip root.

## Environment

The notebooks were written for **Google Colab (Tesla T4) with Google Drive mounted**. Checkpoints, caches and results are saved to Drive, so most notebooks resume or skip finished experiments after a disconnect.

1. Add a Hugging Face token as a Colab secret named **`HF_TOKEN`**. All data comes from the dataset `Exploration-Lab/CS-781-Capstone`.
2. Edit the Drive paths in each notebook's config cell to point at your Phase-1 checkpoints and `final_bpe_en.json`.
3. Run Phase 1 first, then any later phase in any order.

Main dependencies: `torch`, `ijson`, `huggingface_hub`, `datasets`, `pyarrow`, `scikit-learn`, `rouge-score`, `bert-score`, `sacrebleu`, and optionally `mamba-ssm` for the fused Mamba scan kernel (Phase 5).

## Known issues found while documenting

These come straight from the saved notebook outputs and are worth fixing before the results are reported:

| Where | Issue |
|---|---|
| Phase 1, GQLA | The training loss shifts the labels twice, so the model learned to predict the token **two** positions ahead. This explains test ROUGE-L ≈ 0.002. |
| Phase 1, TXL | The config cell shows the *large* run (32 layers, d=512), which **diverged** (val ppl ≈ 1,076). Later phases use the *small* checkpoint (8 blocks, d=768). |
| Phase 4, Enc–Dec | `BEST_REP` overrode `method`, so every Part-B run (full, partial, gradual, adapter, prompt…) silently trained the **frozen** model. Their scores are identical. |
| Phase 4, TXL | Every run collapsed to a single class (macro-F1 0.333, loss ≈ ln 2). |
| Phase 6, TXL | Loads `transformer_xl_gqa-large.pt`, the diverged checkpoint, instead of `…-small.pt`. |
| Phase 7, TXL | Zero-shot and fine-tuned dense retrieval are near random (F1@7 ≈ 0.01–0.03, below BM25). |
| Phase 5, Mamba / TXL | Several runs are incomplete. Mamba trained on 4k of 42.75k documents with the fused kernel off. AdaLoRA logs report rank 0. |
| Phases 3 and 7 | No Mamba notebook. Coverage of all four architectures per phase is incomplete. |

## Cross-phase takeaways

- **GQLA is the most reliable downstream backbone.** It is best in Phases 3, 4, 5 and 7, despite its Phase-1 generation bug. It is used as an adapted encoder, so its next-token behaviour matters less.
- **Mamba is the best generator** (Phase 1) and the **best NER model** (Phase 2). It is also the slowest to fine-tune in pure PyTorch, which limited Phases 4 and 5.
- **The encoder–decoder is the natural summariser.** Fusion-in-Decoder gives the best ROUGE in Phase 6.
- **PEFT usually matches full fine-tuning** (Phases 2, 3, 5, 6, 7) with 1–5% of the trainable parameters. The exception is judgment prediction (Phase 4), where full FT wins clearly.
- **KD works when the student starts from Phase-1 layers.** It is decisive in Phase 4 (GQLA: 0.664 with KD vs 0.339 without), and in Phase 6 the GQLA student beats its teacher.
