# Dual-scale MXFP4 GEMM handoff

Date: 2026-08-16

## Executive status

The dual-scale implementation now has a compiler-viable correctness path, but
it does **not** yet meet the performance goal of matching the pure MXFP4
kernel. The public `dualscale_matmul` entry point is intentionally wired to
the sequential correctness implementation while the faster
warp-specialized implementation remains experimental.

Measured on GPU 4 with the local Triton environment:

| Path | Shape/configuration | Result |
|---|---|---:|
| Pure MXFP4 baseline | `M=N=K=8192`, tuned 2-CTA pipeline | ~6498 TFLOPS |
| 90% target | same baseline | ~5849 TFLOPS |
| Dual-scale sequential path | `M=N=4096, K=8192`, 2 CTA | ~8.3 TFLOPS |

The large gap is expected from the current fallback: it serializes load,
MMA, coarse-scale application, and output for every coarse block. No claim of
90% performance should be made until a pipelined implementation is measured
on the same shape and configuration as the baseline.

## Relevant files

- `07-mxfp4-matmul.py` — pure MXFP4 reference and tuned baseline.
- `08-dualscale-mxfp4-matmul.py` — dual-scale quantization, reference path,
  experimental fused path, and public sequential fallback.
- `dualscale_mxfp4_background.md` — MXFP4 quantization and dual-scale design
  background.

## Mathematical data flow

For a quantized value in a 32-element fine-scale group:

```text
x[q] ~= q_fp4 * fine_e8m0[group32] * coarse_fp32[group512]
```

For GEMM, the K dimension is split into 512-element coarse blocks. For each
coarse block `c`:

```text
partial_c = MMA_scaled(QA_c, QB_c, fine_scales)
C += (coarse_A_c outer coarse_B_c) * partial_c
```

The coarse factor is a row/column outer product. It is therefore not a single
scalar that can be folded into the MMA accumulator for a general tile.

## Current correctness implementation

`dualscale_sequential_kernel` performs the following operations:

```text
running = zero FP32 tile

for c in coarse_blocks:
    load MXFP4 A/B data for K block 512
    load fine E8M0 scales
    partial = tcgen05_mma_scaled(..., use_acc=False)

    load coarse_A[:, c]
    load coarse_B[:, c]
    partial = partial.load()
    running += partial * coarse_A[:, c][:, None] * coarse_B[:, c][None, :]

store running
```

This uses one FP32 partial TMEM allocation and keeps the accumulated result in
registers. It avoids the need for a second large TMEM accumulator and has been
checked for both one-CTA and two-CTA layouts.

The public launcher currently calls this path. The partitioned
`dualscale_kernel` is retained as an experimental development path; its
epilogue is not the validated public implementation.

## Why `use_acc=True` cannot directly solve the coarse-scale step

For one coarse block, `use_acc=True` computes conceptually:

```text
old_accumulator + partial_c
```

The required operation is:

```text
old_running + coarse_A_c * coarse_B_c * partial_c
```

Since the coarse factor varies by output row and column, the old accumulator
would need to be multiplied before the next accumulation. Updating the MMA
partial in place is not equivalent: it would either lose the previous running
result or apply the coarse factor at the wrong point. An in-place design needs
another live state, a spill/reload protocol, or an approximation that folds
the coarse factor into the fine scales.

## Validation already performed

The sequential path compiled and produced finite results for these cases:

- 1 CTA, `M=N=128`, `K=512`: maximum absolute error about `0.031`.
- 1 CTA, `M=N=128`, `K=1024`: maximum absolute error about `0.031`.
- 2 CTA, `M=N=256`, `K=512`: maximum absolute error about `0.031`.
- 2 CTA, `M=N=256`, `K=1024`: maximum absolute error about `0.062`.

The error is consistent with the FP16 output/rounding behavior of the example
and is not a proof of production-level numerical tolerance. Any future kernel
must compare against `dualscale_matmul_reference` and report the exact shape,
dtype, and tolerance.

Useful checks:

```bash
cd /dev/shm/.tmp_yi/workspace/triton
source .venv/bin/activate
python3 -m py_compile python/examples/gluon/08-dualscale-mxfp4-matmul.py
```

## Main performance blockers

1. The fallback is fully serialized across K blocks.
2. The warp-specialized fused version has not yet been validated end to end;
   several variants caused very long compiler lowering times.
3. Two FP32 tile accumulators exceed the available TMEM budget for the current
   `BLOCK_M=BLOCK_N=256` configuration. A prior design required 576 TMEM
   columns while the hardware limit is 512.
4. Staging arbitrary FP32 coarse vectors in TMEM has layout constraints: a
   copied row needs at least 128 contiguous bits. Expanding the staging shape
   made isolated experiments compile, but the full fused kernel still lowered
   too slowly.
5. Two-CTA arithmetic must use the exact layout of the loaded partial. The
   working pattern derives row/column `SliceLayout`s from
   `partial_reg.type.layout`, expands the vectors, and converts them back to
   that layout before multiplication.

## Recommended next implementation sequence

1. Keep `dualscale_sequential_kernel` as the correctness oracle and do not
   remove it while optimizing.
2. Start from the proven pipeline in `07-mxfp4-matmul.py` and add only one
   change at a time: a 512-wide coarse-block loop, a partial-ready barrier,
   and coarse-scale application.
3. Keep coarse scales in shared memory or registers first; avoid FP32 coarse
   vector TMEM staging until the pipeline is compiler-stable.
4. Preserve one partial TMEM tile initially. If overlap requires a second
   state, reduce tile size, use a lower-precision running state only with an
   explicit accuracy check, or add a specialized lowering rather than silently
   changing the math.
5. Compile and validate `K=512`, then `K=1024`, before attempting `K=8192`.
6. Benchmark pure and dual paths back-to-back at identical `M`, `N`, `K`,
   tile, CTA, and buffer settings on an otherwise idle GPU.
7. Compare against the target using:

   ```text
   dual_tflops / pure_tflops >= 0.90
   ```

## Handoff invariants

- Coarse scales have shape `[output_dimension, K / 512]`.
- Fine scales remain E8M0 and retain the baseline MXFP4 packing/layout.
- Coarse scale application occurs once per 512-wide K block.
- No `use_acc=True` accumulation crosses a coarse-scale boundary unless the
  implementation explicitly preserves the pre-scaled running state.
- The sequential path remains available for correctness debugging.
- Performance results must include the exact GPU, matrix shape, and kernel
  configuration.

