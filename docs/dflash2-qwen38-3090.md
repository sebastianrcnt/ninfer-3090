# DFlash 2 speculative decoding for Qwen3.8-27B on RTX 3090

Status: DFlash 2 is functionally complete for the RTX 3090 single-user target.  The real 27B
artifact now passes a 64-token, three-prompt greedy identity test with draft length seven for
both INT8 and TurboQuant KV.  The earlier 103.5 tok/s GSM8K and 94.2 tok/s HumanEval numbers were
measured on the faster pre-closure small-T path; they must not be presented as throughput of the
final bit-identical path.

TurboQuant extension status: implemented and serving at 262K capacity.  Mixed Korean, English,
code, and arithmetic retrieval passes at 31.5K and 97.9K on the final exact build.  The strict
no-throughput-regression contract remains open:
prefill is still below INT8, and exact small-T verification also reduced observed decode from
32.3 to 24.3--26.0 tok/s at 31.5K and from 17.2 to 15.2 tok/s at 97.9K.  Do not read functional
completion as completion of that throughput criterion.

Hardware: one RTX 3090 (GA102, `sm_86`, 24 GiB, 936 GB/s), driver 590.48.01, CUDA 13.1.
Target artifact: `qwen3_8_27b.ninfer`, 18,210,531,328 B on disk, 16.67 GiB resident.
Engine: `/build/apps/ninfer-serve` in `ninfer-3090:dev`, INT8 KV, CUDA Graphs on,
`--max-concurrency 1`.

## MTP-3 baseline

Measured through the serving path (`POST /v1/chat/completions`) against the production
configuration, 2 warmup + 5 measured iterations per case. Acceptance and decode rate are
the engine's own per-request values; round time is derived as
`1000 x tokens_per_round / decode_tok_s`.

### tg128, production sampling (temperature 1.0, top-p 0.95, top-k 20)

| Case | decode tok/s | tokens/round | round ms | accept |
|---|---:|---:|---:|---:|
| GSM8K | 78.9 | 3.29 | 41.7 | 76.5% |
| HumanEval | 64.4 | 2.69 | 41.8 | 56.7% |
| MT-Bench | 53.7 | 2.25 | 41.9 | 42.3% |
| mean | 65.7 | 2.74 | 41.8 | 58.5% |

### tg128 greedy / tg512 sampling

| Case | greedy tok/s | greedy tok/round | tg512 tok/s | tg512 tok/round | tg512 round ms |
|---|---:|---:|---:|---:|---:|
| GSM8K | 86.5 | 3.64 | 72.8 | 3.14 | 43.1 |
| HumanEval | 73.6 | 3.15 | 58.4 | 2.52 | 43.1 |
| MT-Bench | 58.8 | 2.49 | 49.5 | 2.12 | 42.9 |

Peak GPU memory across every baseline run: 22,718 MiB.

**Round time is invariant at 41.7-43.1 ms across task, sampling mode, and decode length.**
Throughput varies only through acceptance length. Any throughput target therefore
decomposes into exactly two independent factors: tokens accepted per round, and round time.

## Speculative cost structure

Same engine, same artifact, varying only the speculative backend:

| Configuration | decode tok/s | tokens/round | round ms | drafting cost |
|---|---:|---:|---:|---:|
| no `--spec` | 35.9 | 1.00 | 27.9 | — |
| `--spec mtp --draft-tokens 3` | 87.0 | 3.64 | 41.8 | 13.9 ms / 3 tokens = 4.6 ms each |
| `--spec mtp --draft-tokens 5` | 81.8 | 4.57 | 55.9 | 28.0 ms / 5 tokens = 5.6 ms each |

Two facts follow, and together they are the reason DFlash 2 is the right mechanism here:

1. **One target forward pass costs 27.9 ms.** That is the floor for any round, and it is
   87% of the 32 ms round-time target. The 27B weights are 16.67 GiB, so at the 936 GB/s
   specified bandwidth a single streaming read is 19.5 ms; the measured pass is 1.43x that.
2. **MTP's drafting cost is linear in draft length**, roughly 5 ms per draft token, because
   MTP drafts autoregressively. Extending MTP to seven draft tokens would cost about 35 ms
   of drafting on top of the 27.9 ms verify — a ~63 ms round. Even at the model card's MTP
   acceptance of 5.02 that is only ~80 tok/s. MTP-5 already demonstrates the effect
   directly: higher acceptance than MTP-3 (4.57 vs 3.64) but *lower* throughput.

DFlash 2 drafts a whole block of eight in a single drafter pass, so its drafting cost is one
forward pass of a 1.92B drafter rather than seven passes of anything.

## Pre-implementation throughput projection (superseded)

Drafting cost estimate: 1.90 GiB of W8 drafter weights at 936 GB/s is 2.2 ms of streaming;
allow 3-4 ms for the pass including the two-tap convolutions, plus ~1 ms for the candidate
selector (a top-16 over 8 x 248,320 logits and the path walk). Verify covers eight positions
instead of one; that adds 0.43 TFLOP of compute, which at ~71 TFLOPS BF16 dense is ~6 ms and
must be hidden behind the 27.9 ms of weight streaming rather than added to it.

Round time target: **32-33 ms**.

Acceptance lengths are from the model card (measured on one H200, block size 8):

| Case | DFlash 2 acceptance | projected tok/s at 32.5 ms | MTP-3 measured | speedup |
|---|---:|---:|---:|---:|
| GSM8K | 5.46 | 168 | 78.9 | 2.13x |
| HumanEval | 4.39 | 135 | 64.4 | 2.10x |
| MT-Bench | 4.10 | 126 | 53.7 | 2.35x |

This was the estimate used before the kernels existed. It proved optimistic because it assumed
the eight-position verify compute could be hidden behind the target weight stream. The final
fast-candidate result was 103.5 tok/s on GSM8K and 94.2 tok/s on HumanEval; the later ceiling
analysis explains why 130 tok/s is not a credible kernel-only target on this card, and the final
greedy-identity closure above records the additional exactness cost.

## Resident memory decision

MTP and DFlash are mutually exclusive backends, so DFlash 2 inherits the memory MTP releases.
Measured startup plans, all with INT8 KV:

| Configuration | KV capacity | runtime | free after startup | GPU total |
|---|---:|---:|---:|---:|
| `mtp --draft-tokens 3` (production) | 131,072 | 4.94 GiB | 1.38 GiB | 22,718 MiB |
| `mtp --draft-tokens 5` | 131,072 | 4.95 GiB | 1.37 GiB | 22,722 MiB |
| no `--spec` | 131,072 | 4.60 GiB | **2.40 GiB** | 21,670 MiB |
| `mtp --draft-tokens 3` | 98,304 | 3.85 GiB | 2.47 GiB | 21,596 MiB |
| `mtp --draft-tokens 3` | 65,536 | 2.75 GiB | 3.57 GiB | 20,474 MiB |
| `mtp --draft-tokens 3` | 32,768 | 1.66 GiB | 4.66 GiB | 19,352 MiB |

Derived: MTP-3 residency is 1.02 GiB (22,718 - 21,670 MiB). KV costs 35.0 KiB/token
(2.19 GiB per 65,536 tokens), independent of the speculative backend.

Budget at the production KV capacity of 131,072 tokens:

| Item | Size |
|---|---:|
| Available once MTP is released | 2.40 GiB |
| DFlash 2 drafter weights, W8 matrices with BF16 codebooks (measured artifact) | -2.016 GiB |
| Drafter KV: 5 sliding layers x 2,048 window x 8 KV heads x 128 dim x 2, INT8 | -0.02 GiB |
| Target feature buffers: 5 layers x 5,120 hidden x 1,024 prefill chunk, BF16 | -0.05 GiB |
| CUDA Graph allowance and selector tables (estimate) | -0.10 GiB |
| **Remaining** | **~0.22 GiB** |

**The budget closes at the full 131,072-token context without reducing KV capacity.** The
desktop (Xorg, gnome-shell, sunshine) holds a further 356 MiB that is outside the engine's
accounting and is already reflected in the measured GPU totals above.

Reducing the served context to 98,304 tokens releases another 1.09 GiB. That is the designated
fallback lever; it is not needed by the W8 configuration.

It *is* needed to measure the drafter in BF16. An unquantised drafter is 3.584 GiB against
2.40 GiB of available memory, so the W8-versus-BF16 acceptance comparison cannot run at the
production KV capacity. At 65,536 tokens the no-speculation configuration leaves 4.46 GiB, which
holds the BF16 drafter with 0.88 GiB to spare; the comparison has to be read at that context.

## Drafter artifact

The drafter ships as a **sidecar** `.ninfer` container rather than as extra objects inside
`qwen3_8_27b.ninfer`. Two reasons, one of them binding:

* Merging would make the drafter a property of the 16.67 GiB target artifact, so every drafter
  revision would require reconverting weights that did not change.
* Reconversion needs the BF16 `Qwen/Qwen3.8-27B` checkpoint. This deployment holds the converted
  target artifact and a Q4_K_S GGUF, but **not** the BF16 source, so a merged artifact could not
  be produced here at all without re-downloading roughly 54 GiB.

Conversion:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_dflash2 \
  --model <Qwen3.8-27B-DFlash2 checkpoint> \
  --out out/qwen3_8_27b_dflash2.ninfer
```

| Property | Value |
|---|---|
| objects | 66 tensors, from 81 source tensors (q/k/v and gate/up are concatenated) |
| formats | 32 W8G32_F16S matrices; 34 BF16 norms, convolution base kernels, and selector codebooks |
| size | 2,164,136,448 B = 2.016 GiB |
| SHA-256 | `01844213651ae4593dacbf79ef74508212299b2ecc3fc450c9afd2825564a635` |
| identity | `qwen3.8-27b` / `dflash2-groupwise-int`, target key `qwen3_8_27b_dflash2` |
| conversion | 7.8 s on the RTX 3090 |

The converter validates every architectural constant in the checkpoint's `config.json` against the
object contract before writing, including `is_causal: false` — a causal checkpoint would draft a
different distribution without any other visible symptom.

The two selector codebooks are the one deliberate departure from the DFlash 1 format choices. They
are 248,320 x 256 each, but they are *gathered* — at most `block_size x top_k` rows per
verification — not streamed. Quantising them saves no bandwidth while putting error straight into
the scores that choose between candidates, so they stay BF16 at a cost of 127 MiB. That is what
moves the artifact from 1.904 GiB to 2.016 GiB.

## Drafter architecture

`z-lab/Qwen3.8-27B-DFlash2` (mirror of `incoai/Qwen3.8-27B-DFlash2`), `model.safetensors`
3,848,817,896 B BF16, 1,924.4M parameters, Apache-2.0.

| Property | DFlash 2 (Qwen3.8-27B) | DFlash 1 (Qwen3.6-35B-A3B, already in tree) |
|---|---|---|
| drafter layers | 5, all sliding (window 2,048) | 6, 5 local + 1 full |
| hidden / intermediate | 5,120 / 17,408 | 2,048 / 6,144 |
| query / KV heads, head dim | 32 / 8, 128 | 32 / 8, 128 |
| target feature layers | 5: {5, 19, 33, 47, 61} | 8: {1, 6, 11, 16, 22, 27, 32, 37} |
| block size | 8 | 8 |
| mask token | 248,070 | 248,077 |
| intra-block attention | bidirectional (`is_causal: false`) | causal |
| two-tap dynamic convolution | kernel 2, group 16 | absent |
| candidate path selector | rank 256, top-k 16 | absent |

Attention head geometry is identical to the existing DFlash 1 path, so its attention kernels
and W8 row-view GEMM specialization (`require_dflash_row_views`, `problem.k == 2048`) are
reusable once generalized off the hard-coded 2,048 hidden size.

## Reference implementations

All four references named in the task were retrieved and are real:

| Reference | State |
|---|---|
| `sgl-project/sglang#35371` "[Spec] DFlash2: local convolution + candidate selector" | merged, +929/-61 over 7 files |
| `vllm-project/vllm#52816` same title | open, +755/-5 over 11 files |
| `z-lab/dflash` | exists, 5,652 stars |
| model card acceptance/throughput tables | retrieved, reproduced above |

The SGLang PR is the authoritative algorithm source since it is merged. Its
`DFlashGroupedConv` wraps each sublayer: one projection of the sublayer input yields both the
input-side and output-side convolution coefficients, the base kernel is initialized to
identity at tap 0, and the shifted taps are masked by position within the block so the
convolution resets at every block boundary.

## Method

Serving-path measurement, 2 warmup + 5 measured iterations per case, CUDA Graphs enabled,
INT8 KV, one request in flight. Prompts are GSM8K / HumanEval / MT-Bench representatives.
The production Qwen unit was stopped and the benchmark server was the only inference process;
`immich_machine_learning` remained unpaused in its normal state. Peak GPU memory is sampled from
`nvidia-smi` at 200 ms across each run and is a whole-device total, not an engine-reported figure.
No GPU power limit, undervolt, voltage/frequency curve, or clock setting was changed.

## What the engine already provides

The Qwen3.6 family runtime is already parameterised over a DFlash backend; DFlash 1 ships for the
35B-A3B target. The 27B target declares the same interface with the feature switched off
(`DFlashConfig::supported = false`, `kMaximumDFlashDraftTokens = 0`, and a
`dflash_graph_profiles` that returns an empty vector).

Reusable as-is:

| Facility | Location |
|---|---|
| target hidden-state capture at selected layers | `DFlashFeatureSink`, `Config::target_feature_layers` |
| drafter KV (sliding local window + paged full) | `DFlashPersistentState` in `dflash_context.h` |
| non-causal intra-block attention | `ops::bidirectional_gqa_attention` (test passes) |
| masked block preparation | `ops::prepare_masked_block` (test passes) |
| lossless accept/commit | `ops::speculative_round` (test passes) |
| W8 drafter GEMM specialisation | `require_dflash_row_views` in `w8_pair_plan.cpp` |
| `--spec dflash --draft-tokens 1..15` CLI and serving surface | `speculative_options.h`, `serve_options.cpp` |

Because the drafter's attention geometry (32 query heads, 8 KV heads, head dim 128) is identical
to DFlash 1's, the attention path needs generalising off the hard-coded 2,048 hidden size rather
than rewriting. Item 3 of the task — extracting target hidden states without an extra weight read —
is already how `DFlashFeatureSink` works; it needs the 27B feature layers wired, not a new
mechanism.

## Drafter loading

The drafter is bound from its own container, so the load path carries two artifacts:

* `EngineOptions::draft_artifact_path`, set by `--draft-artifact` on both the CLI and the server.
* `targets::construct_target` opens a second `artifact::Reader` and calls `plan_draft_load` before
  KV capacity is resolved, so the drafter's residency is inside the same budget as the target's
  rather than discovered after the fact.
* A target declares `accepts_draft_artifact`. Qwen3.6-35B-A3B carries its DFlash 1 drafter inside
  its own artifact and sets it false, so passing a drafter container to that target is rejected
  rather than silently ignored.
* `--spec dflash` and `--draft-artifact` require each other; neither is accepted alone.

`bind_draft_artifact` binds all 66 `dflash2/*` objects as resident and materialises them into a
second device arena, which `LoadedModelData` holds alongside the target's.

## Operators

Two operators are specific to DFlash 2 and are implemented and validated:

| Operator | Shape at this target |
|---|---|
| `ops::dflash_conv` | two-tap grouped dynamic convolution, hidden 5,120, 320 groups of 16, block 8 |
| `ops::dflash_selector` | top-16 over 248,320 logits per position, then the rank-256 lattice walk |

Both are checked against z-lab/dflash's own PyTorch implementation running on CPU with the
released weights (`tests/ops/test_dflash_conv.cpp`, `tests/ops/test_dflash_selector.cpp`, skipped
unless `NINFER_DFLASH2_ORACLE` points at a dump). Two properties make that comparison meaningful:

* The dump rounds every input, weight, and result to bfloat16, matching what the kernels read and
  write. Against a float32-input reference the convolution missed by exactly one bfloat16 ulp
  (max relative 0.00389 = 2^-8) — the final rounding step, not the arithmetic.
* Selector logits drawn from a continuous distribution are not a well-posed fixture: bfloat16
  keeps 8 mantissa bits, so exact ties are common across a 248k vocabulary and the top-k *set*
  stops being unique. A tie broken differently sends the whole walk down another path. The dump
  separates the leading candidates by far more than a bfloat16 ulp. Tie behaviour is not a
  correctness property of the operator — verification against the target still decides what is
  accepted — so the test compares candidates as a set and matches each score to its own token.

Both pass at a 1e-3 relative criterion.

## Measured DFlash 2 results

The first end-to-end DFlash 2 performance closure used the same serving harness as the baseline:
2 warmup + 5 measured iterations, tg128, greedy, INT8 KV, CUDA Graphs, one request in flight, and
a 32,768-token KV capacity.  Subsequent longer-output testing found that this fast path was not
bit-identical to repeated T=1 target execution, so these figures are retained as an optimization
baseline rather than claimed as the final lossless throughput.

| Case | tokens/round | round ms | accept | server decode tok/s | client tok/s range |
|---|---:|---:|---:|---:|---:|
| GSM8K | **6.06** | 58.6 | 72.3% | **103.5** | 101.2-101.4 |
| HumanEval | **5.52** | 58.6 | 64.6% | **94.2** | 92.7-92.8 |
| MT-Bench | **3.34** | 58.6 | 34.6% | 57.0 | 56.6-56.8 |

This fast candidate met the requested single-user thresholds: GSM8K was 3.5 tok/s over the
100 tok/s target and HumanEval was 14.2 tok/s over the 80 tok/s target. Peak whole-device GPU
memory was 20,966 MiB. Its raw run is
`scratchpad/dflash2/final_q45_small_t_lossless.json` outside the repository.

The pre-kernel DFlash 2 measurement was 87.9 / 73.9 / 48.2 tok/s with 69-72 ms rounds. Relative to
that handoff baseline, the fast candidate was +17.7% / +27.5% / +18.3%, and its round was about
11 ms shorter. Acceptance was not manufactured by the optimization; the speedup came from kernel
time. These deltas predate the final exact SIMT replacement.

### Fast small-T MMA candidate

The performance candidate added two T=2..8 routes:

| Operator | old T=8 | final T=8 | change | implementation |
|---|---:|---:|---:|---|
| Q5 MLP down + residual | 204.8 us | **122.880 us** | -40.0% | 8 K-split warps, 64 rows/CTA, BF16 Tensor Core MMA, residual epilogue |
| Q4 gate/up + SwiGLU | 222.208 us | **136.288 us** | -38.7% | 8 K-split warps, 4 row tiles/CTA, 32 outputs/CTA, SwiGLU epilogue |

Both kernels reuse each staged activation fragment across multiple row tiles. The Q5 kernel keeps
the proven 8-way K reduction and applies each f16 group scale after the integer-Q MMA using the
same `fmaf` accumulation order. Its 16-warp variant reached 114.688 us but changed the reduction
tree and failed the real-model losslessness test, so it was rejected. The Q4 kernel also retains
the original 8-way reduction; lowering its `launch_bounds` minimum from three blocks/SM to two
lets the compiler retain enough registers and changes T=8 from 179.2 to 136.3 us without changing
the arithmetic.

The later real-artifact closure replaced the T=2..8 MMA routes with SIMT kernels that reproduce
the T=1 FMA and reduction order. T=9..16 keeps the existing split-2 route.

### Final greedy-identity closure

The original four-token real test did not exercise enough decode rounds to expose every
width-dependent difference.  The final test generates 64 tokens for three prompt families,
first with ordinary target decoding and then with DFlash 2 at draft length seven.  It runs that
comparison independently with INT8 and TurboQuant KV, requires identical token-id vectors, and
also requires nonzero speculative rounds.

To close the failures without another target-weight read per verified position, the final T=2..8
path streams each Q4/Q5 weight tile once but maintains a separate FP32 accumulator for every
position.  Per position it preserves the T=1 dequantization, FMA sequence, warp reduction, and
epilogue order.  This applies to attention input projection, fused and materialized GDN input
projection, gate/up SwiGLU, down/residual projection, and the output-side residual projections.
The fixed-width TurboQuant attention path likewise matches sequential T=1 for every valid masked
prefix; INT8 width 7--16 uses sequential one-token attention chunks because its split partition
is defined by the chunk-final window.

The cost is measurable.  On the final exact build, the 31,445-token retrieval request measured
755.8 tok/s prefill and 24.3 tok/s decode on its cold request (26.0 tok/s on a cached repeat), while
the 97,945-token request measured 503.6 tok/s prefill and 15.2 tok/s decode.  Both recovered all
four needles.  These are the authoritative final-code numbers; the faster rows elsewhere in this
document are explicitly historical baselines.

The same final code was measured in a separate 32,768-token-capacity TurboQuant+DFlash service
using the original tg128 greedy serving harness (two warmups and five measured requests per case):

| Case | current TurboQuant exact | tokens/round | round ms | historical fast INT8 candidate |
|---|---:|---:|---:|---:|
| GSM8K | **70.4 tok/s** | 6.12 | 87.0 | 103.5 tok/s |
| HumanEval | **60.7 tok/s** | 5.29 | 87.2 | 94.2 tok/s |
| MT-Bench | **46.5 tok/s** | 3.43 | 73.8 | 57.0 tok/s |

Peak whole-device use was 20,356 MiB.  The raw current run is
`scratchpad/dflash2/turboquant_32k_exact_20260820.json` outside the repository.  The historical
column is a different KV representation and is not a strict dtype-to-dtype comparison; it makes
the exact small-T cost visible, not a TurboQuant-only attribution.

### Profiler findings and rejected approaches

Three earlier premises were false and must not be reused:

1. The old W8-versus-Q4/Q5 slope comparison mixed different models. W8 streamed the 35B target's
   25.2 MB geometry, while Q4/Q5 streamed the 27B target's 52.4 MB geometry. It was not an
   apples-to-apples proof that the Q4/Q5 slope could disappear.
2. Dequantization ALU was not the bottleneck. `ncu` showed ALU at about 42% while L1/TEX
   transactions were about 90% of the limit, and `Q5SimtDecodeAtom` already uses the integer
   magic-number decode. Pre-substituting fragments and adding another magic-number transform did
   not address the saturated pipe.
3. Q4 scheduling is not the GDN critical path. At the real shape the Q5 side is about 172 us at
   80% compute utilization; the Q4 side is about 143 us and overlaps it through PDL. Q4 schedule
   changes left the combined operator at 264-270 us.

An unscaled Q5 GDN MMA prototype reduced that operator to about 217 us, and multiplying scales
before MMA produced about 218 us. Both changed real-model greedy tokens. They were removed rather
than weakening the losslessness requirement. PDL experiments on attention were also slower
(about 261-278 us versus 199 us cold), so the original sequential launch remains.

### Throughput ceiling

The measured round decomposes into a roughly fixed 29.4 ms target-weight stream plus a
position-dependent term across eight verified positions. The 130 tok/s stretch number would
require about 1.84 ms per position. Even the loop-line floors of the three large operator sets —
GDN 12.7 ms, gate/up 14.2 ms, and down 13.0 ms per eight-position round — already sum to about
1.9 ms per position before the rest of the graph. The 16.67 GiB target stream itself has a
theoretical lower bound near 21 ms on this card.

Consequently a perfect rewrite of every remaining kernel is expected to top out around 120 tok/s,
not 130 tok/s. Reaching 130 requires changing the numerical target, model representation, or
hardware assumption; it is not a credible kernel-only milestone. The requested 100/80 tok/s
contract is achieved without any such change.

## TurboQuant / PolarQuant 262K KV extension

The production server now uses `--kv-dtype turboquant --max-context 262144
--kv-capacity 262144`.  This is a runtime-only KV representation: neither the 18.21 GB target
artifact nor the 2.16 GB DFlash 2 sidecar is converted or modified.  No GPU power, voltage,
clock, or power-limit setting was changed.

### Packed representation

Each 256-dimensional K or V row first receives the same deterministic signed orthogonal
Hadamard rotation.  PolarQuant stores the resulting unit direction as the complete 255-node
binary angle tree at three bits per node and stores one FP16 radius.  Keys additionally store a
256-bit QJL residual sign projection and one FP16 residual norm.

| Plane | Bytes / head / token | Contents |
|---|---:|---|
| V | 98 | 96-byte 3-bit Polar tree + FP16 radius |
| K | 132 | V-format direction + 32-byte QJL signs + FP16 residual norm |
| K + V | **230** | **3.59375 physical bits/value** including both row scales |

The quantized payload itself is 3.5 bits/value; the two FP16 row values account for the
0.09375-bit physical overhead.  Pages remain 64 tokens and are consumed in place.  There is no
persistent BF16 expansion cache and no extra target-weight read.

Across the target's 16 full-attention layers and four KV heads this is 14.375 KiB/token, or
3.59375 GiB at 262,144 tokens.  The preceding INT8 layout measured about 35 KiB/token, which would
be 8.75 GiB at the same capacity.  TurboQuant therefore saves about 5.16 GiB (59%) of target KV
at 262K and raises the deployed capacity from 98,304 to 262,144 tokens (2.67x), while retaining
551.94 MiB of post-startup device headroom on the final launch.

### GPU paths

* Append performs the signed Hadamard, Polar tree quantization, and QJL residual projection
  directly into the paged byte planes.
* T=1..8 uses one fused CTA per KV head/split.  It decodes each packed K/V tile once for every
  query position, reconstructs each cached key's QJL residual once in registers and folds it into
  the Polar key tile, then uses one `sm_86` BF16 Tensor Core QK contraction.  Online softmax/value
  accumulation remains FP32, every fixed width is independently CUDA-Graph capturable, and the
  final route requires no dynamic QJL shared-memory plane.
* Prompt attention is FlashAttention-style with 128 query rows and 64 packed key rows per CTA.
  The QJL residual is reconstructed once per cached key and folded into the transient Polar key,
  replacing the second QJL QK contraction with one BF16 Tensor Core MMA.  A scalar, exact
  16-coordinate tree decoder preserves each coordinate's multiplication order without the old
  dynamically indexed `float[16]` local stack; centroid tables reside in device constant memory.
  The two children required at levels 1--4 are stored as adjacent, exact `(cos, sin)` pairs, so
  each tree step uses one 64-bit constant lookup instead of two divergent scalar lookups.
  Query-head is the fastest grid axis so the six GQA siblings reuse nearby packed KV cache lines.
  The final T=128 kernel is about 126 us, and a T=4096 one-layer benchmark improved from 14.83 to
  14.43 ms after the cache-local grid ordering, then to 11.53--11.56 ms after paired centroid
  loads.  T=8192 similarly improved from 51.14 to 41.51--42.32 ms.
* INT8/BF16 dispatch and their kernels are unchanged and remain available as fallback modes.

The DFlash 2 artifact has five local-attention layers and no full-attention drafter layer.  The
old runtime nevertheless reserved a context-sized BF16 full-KV pool for their common ABI.  The
planner now retains one dummy page for all-local DFlash while preserving the original pool for
DFlash variants that actually have full-attention layers.  This is what closes the 262K memory
budget without weakening the target cache.

### Production memory

Final service parameters include `--prefill-chunk 2048`, CUDA Graphs, DFlash 2 with seven draft
tokens, and one request in flight.  Startup reports:

| Item | Final value |
|---|---:|
| resolved KV capacity | 262,144 tokens / 4,096 pages |
| runtime reservation | 4.46 GiB |
| free after target weights | 4.98 GiB |
| free after startup | **551.94 MiB** |
| allocator slack | 533.53 MiB |
| CUDA Graph use / allowance | 12.00 / 32.00 MiB |

This exceeds the required 300 MiB device headroom while the desktop and Sunshine remain in their
normal state.

### Verification

| Check | Result |
|---|---|
| packed A1 vs FP64 causal attention | relative L2 0.231507 |
| 128-token, two-page prompt vs FP64 causal attention | relative L2 0.234214 |
| append determinism and A1/A3 packed-cache parity | bit-identical |
| CUDA Graph replay, every T=1..8 | bit-identical to eager |
| logical page 4,095 / position 262,143 append + decode | finite output, pass |
| ordinary BF16/INT8 GQA public contract | pass |
| Qwen runtime mechanism checks | pass |
| real target + real DFlash 2 greedy losslessness | **PASS**, 64 tokens x three prompt families x INT8/TurboQuant KV |

The final complete 88-entry suite took 109.99 seconds and retains exactly the eight documented
`sm_86`/NVFP4 and 1e-5 baseline failures; TurboQuant adds no regression.  Eight optional/real
artifact tests skip without their environment variables; the real DFlash 2 test was also run
separately with both artifacts and passed.

### Serving-path context validation

The reproducible harness `tools/bench/run_long_context_retrieval.py` distributes independent
Korean, English, Python-expression, and arithmetic needles through the prompt.  Greedy generation
with thinking disabled recovered all four values exactly at every measured extent:
`해오라기-7319`, `cobalt-orbit-4821`, `x*x+1`, and `1591`.  The following table is the
pre-greedy-identity performance baseline and is retained for direct comparison:

| Input context | Result | prefill | decode | TTFT | wall | DFlash |
|---:|---|---:|---:|---:|---:|---:|
| 31,445 | all four exact | **757.1 tok/s** | **32.3 tok/s** | 41.58 s | 42.75 s | 3.55 tok/round |
| 97,945 | all four exact | **505.2 tok/s** | **17.2 tok/s** | 194.00 s | 196.21 s | 3.90 tok/round |
| 261,953 | all four exact | **242.0 tok/s** | **7.3 tok/s** | 1083.00 s | 1088.17 s | 3.25 tok/round |

The final exact build was then revalidated through the same live 262K service:

| Input context | Result | prefill | decode | TTFT | wall | DFlash |
|---:|---|---:|---:|---:|---:|---:|
| 31,445 | all four exact | **755.8 tok/s** | **24.3 tok/s** cold / **26.0** cached | 41.65 s | 42.93 s | 3.10 cold / 3.55 cached |
| 97,945 | all four exact | **503.6 tok/s** | **15.2 tok/s** | 194.64 s | 197.14 s | 3.90 tok/round |

For 31,445 tokens the cold request used a 32-token output cap and stopped immediately before the
last value; an identical cached repeat with a 64-token cap recovered all four.  The table combines
the cold request's prefill statistic with both observed decode rates and states that distinction
explicitly.

The same 31.5K harness on INT8 KV measured 839.7 tok/s prefill and 32.8 tok/s decode.  The packed
prompt path therefore remains about 10% behind in prefill.  Before greedy-identity closure its
decode was only 1.5% behind and occasionally measured 34.6 tok/s, but the authoritative final
exact measurements above show a larger decode gap. Relative to the earlier stack-free TurboQuant
implementation, Br=128 raised 31.5K prefill from 575.4 to 757.1 tok/s and 97.9K prefill from 311.4
to 505.2 tok/s; the final exact build retains essentially all of that prompt-side gain.

A separate pre-closure short-context stability request generated exactly 512 tokens and stopped
at the output limit: 39 input tokens, 128.5 tok/s decode, 7.97 tok/round (99.6% acceptance), and
4.17 s wall time. It confirms graph and long-generation stability but is not a final exact-path
throughput claim.

Rejected prompt experiments are recorded so they are not repeated: Bc=32 was slower; CTA-wide and
warp-private packed staging reduced uncoalesced transactions but increased spills and latency;
sequential low-register QK loads were neutral; pairing two query heads at 64 rows preserved the
same 128-row reuse and was slightly slower; and reconstruct-then-INT8 QK added 11-12% latency.

### Remaining work

Functional DFlash 2 and TurboQuant capacity are complete, but the throughput contract is not.
TurboQuant prompt attention must close the 31.5K prefill gap to INT8, and the exact small-T target
path must recover the decode regression quantified above. Capacity, retrieval quality, CUDA Graph
stability, and DFlash greedy identity are closed. Q4-only GDN scheduling and extra dequantization
magic-number work are known dead ends for the separate short-context ceiling described above.

## Verification status

Final `ctest` on this build (`sm_86`, CUDA 13.1): **88 entries: 72 passed, 8 skipped, and the
same 8 baseline failures**, 109.99 s. No new failure was introduced. The suite was run in the
CUDA container with the source tree mounted at `/src` and one test process at a time.

| Added test | Result |
|---|---|
| `ninfer_dflash_conv_test` | passes against the z-lab/dflash oracle; skips without the oracle dump |
| `ninfer_dflash_selector_test` | passes against the z-lab/dflash oracle; skips without the oracle dump |
| `ninfer_qwen3_8_27b_dflash2_real_test` | **PASS** with both real artifacts; skips in plain `ctest` |

Losslessness is the important one. It decodes three prompts greedily with no speculative backend,
then the same three with DFlash 2, and requires the generated token ids to be identical. It also
fails if DFlash 2 reported zero speculative rounds, so the comparison cannot pass by never
speculating. Run it with:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=<target>.ninfer \
NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS=<drafter>.ninfer \
  build/tests/ninfer_qwen3_8_27b_dflash2_real_test
```

The eight standing failures are unchanged and unrelated. Tests #72, #74-#77, #83, and #87 reject
or abort on the Blackwell-only NVFP4 A4 / `sm_120a` format, and
`ninfer_gdn_gating_proj_test` (#46) misses its reduction criterion by about 1e-5. The repository's
declared upstream toolchain is RTX 5090 / `sm_120a`.

MTP and the ordinary decode path are untouched: DFlash 2 branches on
`DFlashConfig::convolution` and `DFlashConfig::selector`, both false for Qwen3.6-35B-A3B, whose
DFlash 1 schedule, fused projections, and graph allowance are unchanged.

## Build

Development builds run in a CUDA 13.1.2 container with the source bind-mounted, the build tree on
the system SSD, and ccache on the 2 TB spindle, so an edit/compile cycle does not repeat a full
CUDA configure. `NVCC_PREPEND_FLAGS=--split-compile=6` parallelizes device optimization inside the
large CUDA translation units; the prompt TU rebuild fell to roughly 47 seconds on this host. Set
`CUDA_SPLIT_COMPILE=1` in the local `devbuild.sh` to disable it for compiler diagnostics. The
shipping `Dockerfile` is unchanged.

```bash
cmake -S /src -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
```

The final full incremental `all` build completed without errors and produced `ninfer`,
`ninfer-serve`, all tests, and all benchmarks.

Conversion needs `torch` and `safetensors`, which the system interpreter does not carry; the run
recorded above used an existing local environment (torch 2.11/2.8) read-only rather than
installing into one.
