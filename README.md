<div align="center" id="redknottop">
  <img src="RedKnot_Logo.png" alt="RedKnot logo" width="600" />

  <h1>RedKnot</h1>
  <p><strong>Head-aware reuse and token-selective execution for long-context LLM serving.</strong></p>

  <p>
    <a href="https://github.com/sgl-project/sglang"><img src="https://img.shields.io/badge/built%20on-SGLang-blue" alt="Built on SGLang" /></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="Apache-2.0 License" /></a>
    <a href="#deepseek-v4-flash-release"><img src="https://img.shields.io/badge/DeepSeek--V4--Flash-TP8%20release-orange" alt="DeepSeek V4 Flash TP8 release" /></a>
  </p>

  <p>
    <a href="#deepseek-v4-flash-release">Quick start</a> ·
    <a href="#how-redknot-works">Architecture</a> ·
    <a href="#measurement-protocol">Measurement</a> ·
    <a href="#documentation">Documentation</a> ·
    <a href="https://arxiv.org/abs/2606.06256">Paper</a>
  </p>
</div>

> **Research release.** RedKnot is an SGLang-based research system for long-context inference. Its kernels, policy contracts and supported models are actively evolving. Reproduce the frozen benchmark manifests before using a reported result to compare systems or hardware.

## Performance at a glance

On qualified long-context profiles, RedKnot is designed to keep quality regression
within **1 percentage point** while converting reusable attention and token-level
work into a **2–5× hot-state TTFT speedup** and **70–90% arithmetic compute-ledger
saving**. These are an operating envelope, not a universal guarantee: the
achieved point depends on the model, context length, GPU topology and frozen
policy. The per-suite result JSON is the source of truth.

```text
Qualified long-context operating envelope

Quality regression      ≤ 1 pp   |█                                       |
Hot-state TTFT          2×–5×    |████████████████████████████████        |
Compute-ledger saving   70%–90%  |███████████████████████████████████     |
```

The compute ledger intentionally excludes memory traffic, kernel-launch cost,
TP communication and all uncredited runtime components; it is therefore not a
claim about total system energy or universal end-to-end throughput.

## News

- **2026-08 — DeepSeek V4 Flash TP8 release.** This repository now includes a packaged DeepSeek-V4-Flash + RedKnot path with one-command reproduction over frozen 64K, 128K, 256K and 440K LongBench-derived RAG suites.
- **2026-06 — Paper.** [*RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention*](https://arxiv.org/abs/2606.06256) is available on arXiv.

## What is RedKnot?

Long-context prefill is often dominated by work that is repeated across related documents and by computation applied uniformly to tokens with very different importance. RedKnot exposes that structure to the serving runtime:

| Mechanism | What it does |
|---|---|
| **Head-aware MLA reuse** | Stores reusable local-head MLA artifacts offline, recomputes global/recovery heads online, then merges them in the model projection path. |
| **RoPE-aware segment relocation** | Builds reusable documents in their own canonical position space and restores their position-dependent representation online. |
| **Indexer-guided token-row recovery** | Restricts expensive recovery work to selected transformer rows while retaining the required attention state. |
| **Adaptive sparse MoE Top-K** | Uses a policy-controlled expert budget rather than applying one fixed expert count to every token. |
| **SegPagedAttention runtime** | Represents KV visibility and storage at head/segment granularity instead of requiring one uniform cache policy. |

The design is intended to turn algorithmic reuse into a real serving gain while keeping the reference path explicit and auditable. It is not a prefix-cache benchmark: the release reference is a full online recomputation on the same checkpoint and input IDs.

## DeepSeek V4 Flash release

The DeepSeek-V4-Flash release is the primary reproducible path in this repository. It runs on a TP8 server and ships all frozen inputs, head policy, sparse-MoE policy and execution manifests required for the packaged benchmark.

### Quick start

```bash
git clone https://code.devops.xiaohongshu.com/liuyang52/redknot-0.1.git
cd redknot-0.1/test/srt/redknot

# Creates or validates the pinned environment, then runs all four suites.
./run_deepseek_v4_flash_reproduction.sh
```

The wrapper uses the local DeepSeek-V4-Flash checkpoint by default. Set `REDKNOT_MODEL_PATH` or pass `--model-path` to select another checkpoint path. If the checkpoint is unavailable, the Python entrypoint can download the published model unless `--no-download-model` is set.

For a new shell on a prepared machine:

```bash
cd test/srt/redknot
./setup_deepseek_v4_flash_env.sh --check-only
source ./environment-deepseek-v4-flash.env
python benchmark_RedKnot_DeepSeekV4Flash.py
```

The default run is intentionally comprehensive and sequential: it needs the same eight GPUs for each suite and does not run two TP8 servers concurrently.

### Packaged suites

| Context length | Cases | Contents |
|---|---:|---|
| 64K | 15 | 10 short-answer + 5 long-answer frozen cases |
| 128K | 15 | 10 short-answer + 5 long-answer frozen cases |
| 256K | 15 | 10 short-answer + 5 long-answer frozen cases |
| 440K | 15 | 10 short-answer + 5 long-answer frozen cases |

The suite order and SHA256 digests are frozen in [`test/srt/redknot/datasets/LongBench/suites/RELEASE_SUITES.json`](test/srt/redknot/datasets/LongBench/suites/RELEASE_SUITES.json). The launcher validates this manifest before allocating a GPU.

### What the release reports

For every case, the release writes a side-by-side complete-text comparison:

- **Recomputed** — the same DeepSeek-V4-Flash checkpoint performs a complete online prefill with no RedKnot reuse and without treating document 1 as a prefix.
- **RedKnot** — document 1 is materialized as a certified prefix; subsequent documents use the head-reuse, online recovery and merge path defined by the frozen profile.

It also reports hot-state streaming TTFT and the conservative compute ledger. The default TTFT protocol is three unmeasured warmup pairs followed by ten measured Recomputed/RedKnot pairs, with p50 and p95 reported. Model loading, offline snapshot construction and first-use compilation are not charged to online TTFT.

The compute ledger is deliberately narrower than total system cost: it does **not** give saving credit for Indexer, compressor, router, normalization, memory traffic, kernel launch or TP communication. The default release does not claim a QPS result; run the explicit QPS diagnostic only when its runtime evidence is valid for the requested concurrency.

For environment details, resume semantics, model validation, result layout and the exact metric contract, see the [DeepSeek V4 Flash release guide](test/srt/redknot/README_DEEPSEEK_V4_FLASH.md).

## Other benchmark entrypoints

Alongside the DeepSeek-V4-Flash release path, the repository contains
model-specific RedKnot benchmark entrypoints for Mistral, Qwen and Llama:

| Family | Entry point | Status |
|---|---|---|
| Mistral | `benchmark_RedKnot_Mistral_RAG.py` | Native-SWA reuse benchmark |
| Qwen3 | `benchmark_RedKnot_Qwen3_RAG.py` | Head-aware RAG benchmark |
| Qwen3.5 MoE | `benchmark_RedKnot_Qwen35_RAG.py` | MoE benchmark; requires the pinned Transformers 5 environment |
| Llama 3.3 | `benchmark_RedKnot_Llama3.3_RAG.py` | Experimental; validate its model-specific result contract |

Run them from the release directory after installing the required model weights:

```bash
cd test/srt/redknot

# Mistral and Qwen3
python benchmark_RedKnot_Mistral_RAG.py
python benchmark_RedKnot_Qwen3_RAG.py

# Qwen3.5 MoE: use the pinned Transformers 5 environment
../../.venv_tf5/bin/python benchmark_RedKnot_Qwen35_RAG.py

# Llama 3.3: experimental path
python benchmark_RedKnot_Llama3.3_RAG.py
```

Each script owns its model-specific configuration, dataset and hardware
requirements. Do not compare their numbers directly with the DeepSeek V4 Flash
TP8 release unless their reported input, precision and measurement contract
match.

## How RedKnot works

```text
Offline document preparation
  document tokens ──► head-aware MLA artifacts + shared latent/cache state
                                  │
                                  ▼
Online RAG prefill
  query + documents ──► global/recovery heads online ──► projection merge ──► output
                             │                    ▲
                             └─ Indexer-selected rows / adaptive MoE Top-K
```

For the DeepSeek V4 Flash profile, dense boundary layers remain online. In the middle layers, the frozen head policy separates online global heads from reusable local heads. The runtime validates policy identity, segment geometry, artifact provenance and restoration evidence before it accepts reuse.

This separation matters: a lower arithmetic compute count alone does not prove a lower end-to-end latency. The release therefore publishes both compute-ledger and client-observed TTFT measurements.

## Measurement protocol

Reproducibility and honest comparison are first-class requirements:

1. Recomputed and RedKnot receive identical frozen input token IDs and sampling parameters.
2. Recomputed performs full online computation; it does not use a prefix cache.
3. RedKnot runtime evidence must cover the expected restore forwards and layers; missing or fallback evidence fails closed.
4. Short-answer and long-answer presentation cases are reported separately. The five long-output cases are not silently included in the primary short-answer aggregate.
5. Result JSON files retain machine-readable runtime evidence, while `comparison.md` presents the complete generated text side by side.

Published numbers are specific to the frozen suite, model revision, TP topology and hardware. They should not be read as a universal speedup guarantee.

## Repository layout

```text
python/sglang/srt/layers/attention/redknot/   RedKnot runtime integration
test/srt/redknot/                             Benchmarks, release launcher and docs
test/srt/redknot/head_class/                  Frozen head-policy publication
test/srt/redknot/sparse_ffn_params/           Sparse-MoE policy publication
test/srt/redknot/datasets/                    LongBench inputs, suites and provenance
test/srt/redknot/server/                      TP8 server launcher and policy checks
```

## Model and feature status

| Area | Status |
|---|---|
| DeepSeek-V4-Flash TP8 packaged reproduction | **Release path** |
| Head-aware MLA reuse, RoPE relocation, Indexer recovery | **Release path for the frozen DeepSeek profile** |
| Adaptive sparse MoE Top-K | **Policy-controlled research feature** |
| Other model benchmark scripts | Experimental; validate their own model-specific README and result contract |
| Ascend NPU adaptation | Work in progress |

## Documentation

- [DeepSeek V4 Flash release and reproduction guide](test/srt/redknot/README_DEEPSEEK_V4_FLASH.md)
- [Frozen release suite manifest](test/srt/redknot/datasets/LongBench/suites/RELEASE_SUITES.json)
- [Dataset provenance](test/srt/redknot/datasets/LongBench/PROVENANCE.json)
- [Head-policy contract](test/srt/redknot/head_class/deepseek_v4_flash_0731_redknot.json)
- [Sparse-MoE policy contract](test/srt/redknot/sparse_ffn_params/deepseek_v4_flash_0731.json)
- [SGLang installation guide](https://docs.sglang.io/get_started/install.html)

## Citation

If you use RedKnot, please cite the paper:

> Yang Liu, ZhaoKai Luo, HuaYi Jin, ZhiYong Wang, RuoZhou He, BoYu Wang, Guanjie Chen, and Junhao Hu. *RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention.* [arXiv:2606.06256](https://arxiv.org/abs/2606.06256).

## Acknowledgements

RedKnot is built on [SGLang](https://github.com/sgl-project/sglang) and benefits from the broader serving ecosystem, including [vLLM](https://github.com/vllm-project/vllm).

## License

Released under the [Apache License 2.0](LICENSE).
