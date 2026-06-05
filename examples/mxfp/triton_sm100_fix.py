"""
Monkey-patch Triton's NVIDIA compiler to disable `add_optimize_accumulator_init`
on sm_100 (Blackwell).  Import this BEFORE any Triton kernel compilation.

Usage:
    import triton_sm100_fix  # noqa: F401  — just import it
    import triton
    # ... your kernel code ...
"""

import triton._C.libtriton as _triton

_orig = getattr(_triton.passes.ttgpuir, 'add_optimize_accumulator_init', None)
if _orig is not None:
    _triton.passes.ttgpuir.add_optimize_accumulator_init = lambda pm: None
    print("[triton_sm100_fix] Disabled add_optimize_accumulator_init (sm_100 tmem_alloc bug)")
else:
    print("[triton_sm100_fix] add_optimize_accumulator_init not found (already removed or different Triton version)")
