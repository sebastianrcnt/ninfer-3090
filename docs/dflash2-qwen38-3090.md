# DFlash 2 speculative decoding for Qwen3.8-27B on RTX 3090

Status: complete for the RTX 3090 single-user target. The final lossless DFlash 2 path reaches
103.5 tok/s on GSM8K and 94.2 tok/s on HumanEval using kernel changes only.

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
measured result is 103.5 tok/s on GSM8K and 94.2 tok/s on HumanEval; the later ceiling analysis
explains why 130 tok/s is not a credible kernel-only target on this card.

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

DFlash 2 decodes end to end and losslessly. The final measurement uses the same serving harness as
the baseline: 2 warmup + 5 measured iterations, tg128, greedy, INT8 KV, CUDA Graphs, one request in
flight, and a 32,768-token KV capacity.

| Case | tokens/round | round ms | accept | server decode tok/s | client tok/s range |
|---|---:|---:|---:|---:|---:|
| GSM8K | **6.06** | 58.6 | 72.3% | **103.5** | 101.2-101.4 |
| HumanEval | **5.52** | 58.6 | 64.6% | **94.2** | 92.7-92.8 |
| MT-Bench | **3.34** | 58.6 | 34.6% | 57.0 | 56.6-56.8 |

The requested single-user thresholds are therefore met with margin: GSM8K is 3.5 tok/s over the
100 tok/s target and HumanEval is 14.2 tok/s over the 80 tok/s target. Peak whole-device GPU memory
was 20,966 MiB. The final raw run is
`scratchpad/dflash2/final_q45_small_t_lossless.json` outside the repository.

The pre-kernel DFlash 2 measurement was 87.9 / 73.9 / 48.2 tok/s with 69-72 ms rounds. Relative to
that handoff baseline, the final result is +17.7% / +27.5% / +18.3%, and the round is about 11 ms
shorter. Acceptance is not manufactured by the optimization; the speedup comes from kernel time.

### Small-T MMA kernels

The final path adds two exact T=2..8 routes:

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

The Q5 planner routes only the real 5,120x17,408 MLP-down shape at T=2..8 to the new residual
kernel; T=9..16 stays on the existing exact split-2 route. The Q4 specialization is confined to
the 34,816x5,120 fused gate/up shape and T<=8. The wider generic paths are unchanged.

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

### Optional follow-up work

The goal is complete. Remaining research items are not required for the accepted throughput:

1. Improve the Q5 side of GDN while preserving the exact reduction order; Q4-only work cannot move
   its critical path.
2. Investigate MT-Bench acceptance (3.34 tokens/round versus the model card's 4.10).
3. Measure a BF16 drafter at a reduced KV capacity; it does not fit at the production capacity.
4. Re-negotiate any 130 tok/s target before investing in a full operator rewrite.

## Verification status

Final `ctest` on this build (`sm_86`, CUDA 13.1): **86 entries: 70 passed, 8 skipped, and the
same 8 baseline failures**, 106.38 s. No new failure was introduced. The suite was run in the
CUDA container with the source tree mounted at `/src`, `--cap-add SYS_ADMIN`, and user `0:0`.

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

The eight standing failures are unchanged and unrelated. Three direct NVFP4 tests
(`#70`, `#81`, `#85`) reject the Blackwell-only A4 format, four projection tests
(`#72`-`#75`) abort when their NVFP4 coverage reaches the same `sm_120a` requirement, and
`ninfer_gdn_gating_proj_test` (`#45`) misses its reduction criterion by about 1e-5. The
repository's declared toolchain is RTX 5090 / `sm_120a`.

MTP and the ordinary decode path are untouched: DFlash 2 branches on
`DFlashConfig::convolution` and `DFlashConfig::selector`, both false for Qwen3.6-35B-A3B, whose
DFlash 1 schedule, fused projections, and graph allowance are unchanged.

## Build

Development builds run in a CUDA 13.1.2 container with the source bind-mounted, the build tree on
the system SSD, and ccache on the 2 TB spindle, so an edit/compile cycle does not repeat a full
CUDA configure. The shipping `Dockerfile` is unchanged.

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
