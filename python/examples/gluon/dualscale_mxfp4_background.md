# DualScale MXFP4: Background, Idea, and Formulation

## 1. Background: Standard MXFP4

MXFP4 is a microscaling FP4 format designed to reduce memory and compute cost while preserving more accuracy than using a single scale for an entire tensor.

A tensor is divided into small groups along the reduction dimension, typically groups of 32 values.

For each group:

- values are quantized to FP4 E2M1;
- one shared E8M0 scale is used for the 32 values.

Conceptually:

\[
x_i \approx q_i \cdot s_g
\]

where:

- \(q_i\) is the FP4 value;
- \(g=\lfloor i/32 \rfloor\);
- \(s_g\) is the shared E8M0 scale for group \(g\).

For a matrix:

\[
X \approx Q \odot S
\]

with one scale shared by every 32 values along \(K\).

---

## 2. Limitation of Standard MXFP4

The main limitation is not only the low precision of FP4 itself.

The scale is also restricted.

E8M0 is effectively a power-of-two scale:

\[
s_g = 2^{e_g}
\]

so a group cannot choose an arbitrary real-valued scale.

For example, if the ideal scale for one FP4 group is:

\[
s_{\text{ideal}} = 0.37
\]

standard MXFP4 may have to choose something like:

\[
0.25
\quad\text{or}\quad
0.5
\]

This mismatch can cause:

- clipping of large values;
- excessive quantization error;
- under-utilization of the FP4 dynamic range;
- more values collapsing toward zero.

This becomes especially important at 4-bit precision because FP4 has very few representable values.

---

## 3. Core Idea of DualScale

DualScale introduces two levels of scaling:

1. a **coarse FP32 scale** shared by a larger block;
2. a **fine E8M0 scale** shared by the normal MXFP4 group.

A common configuration is:

- coarse block: 512 values;
- fine block: 32 values.

Since:

\[
512 = 16 \times 32
\]

one coarse block contains 16 standard MXFP4 groups.

The representation becomes:

\[
\boxed{
x_i
\approx
q_i
\cdot
s^{\text{fine}}_g
\cdot
s^{\text{coarse}}_c
}
\]

where:

\[
g=\left\lfloor \frac{i}{32} \right\rfloor
\]

and

\[
c=\left\lfloor \frac{i}{512} \right\rfloor
\]

with:

- \(q_i\): FP4 E2M1 value;
- \(s^{\text{fine}}_g\): E8M0 scale for a 32-value group;
- \(s^{\text{coarse}}_c\): FP32 scale for a 512-value block.

---

## 4. Effective Scale

The effective scale seen by one 32-value group is:

\[
\boxed{
s^{\text{effective}}_g
=
s^{\text{coarse}}_c
\cdot
s^{\text{fine}}_g
}
\]

The fine scale is still power-of-two:

\[
s^{\text{fine}}_g = 2^{e_g}
\]

but the coarse scale is an arbitrary FP32 value.

Therefore:

\[
\boxed{
s^{\text{effective}}_g
=
s^{\text{coarse}}_c
\cdot
2^{e_g}
}
\]

This is the key numerical benefit.

Standard MXFP4 can only choose scales on a fixed power-of-two grid:

\[
..., 0.125,\ 0.25,\ 0.5,\ 1,\ 2,\ ...
\]

DualScale shifts that grid using an arbitrary FP32 factor.

For example, if:

\[
s^{\text{coarse}}=1.48
\]

then the available effective scales become:

\[
..., 0.185,\ 0.37,\ 0.74,\ 1.48,\ ...
\]

So an ideal scale such as:

\[
0.37
\]

can now be represented exactly as:

\[
1.48 \times 0.25 = 0.37
\]

---

## 5. Intuition

Standard MXFP4:

```text
one scale / 32
      |
      v
power-of-two grid

0.125 ---- 0.25 ---- 0.5 ---- 1.0
                   ^
             ideal value may
             lie between points
```

DualScale:

```text
one FP32 coarse scale / 512
              x
one E8M0 fine scale / 32
              |
              v
shifted power-of-two grid

coarse = 1.48

0.185 ---- 0.37 ---- 0.74 ---- 1.48
             ^
       much closer to the
       desired local scale
```

A useful interpretation is:

> The coarse FP32 scale chooses a good scale family for a 512-value region, while the E8M0 fine scales select different powers of two within that family for each 32-value subgroup.

---

## 6. Quantization View

For a coarse block \(c\), suppose the coarse scale is:

\[
s^{\text{coarse}}_c
\]

The values can first be normalized conceptually:

\[
\tilde{x}_i
=
\frac{x_i}
{s^{\text{coarse}}_c}
\]

Then each 32-value subgroup is quantized with normal MXFP4:

\[
\tilde{x}_i
\approx
q_i
\cdot
s^{\text{fine}}_g
\]

Substituting:

\[
x_i
\approx
q_i
\cdot
s^{\text{fine}}_g
\cdot
s^{\text{coarse}}_c
\]

So DualScale can be viewed as:

```text
original value
     |
     | divide by coarse FP32 scale
     v
normalized value
     |
     | standard MXFP4 quantization
     | FP4 + E8M0 / 32
     v
quantized representation
```

---

## 7. GEMM Formulation

Consider:

\[
Y = A B^T
\]

For DualScale:

\[
A_{m,k}
\approx
Q^A_{m,k}
\cdot
S^A_{1,m,g}
\cdot
S^A_{0,m,c}
\]

\[
B_{n,k}
\approx
Q^B_{n,k}
\cdot
S^B_{1,n,g}
\cdot
S^B_{0,n,c}
\]

where:

\[
g=\left\lfloor\frac{k}{32}\right\rfloor,
\qquad
c=\left\lfloor\frac{k}{512}\right\rfloor
\]

Then:

\[
Y_{m,n}
\approx
\sum_k
Q^A_{m,k}
Q^B_{n,k}
S^A_{1,m,g}
S^B_{1,n,g}
S^A_{0,m,c}
S^B_{0,n,c}
\]

Because the coarse scales are constant inside one 512-element K block, this can be grouped as:

\[
\boxed{
Y_{m,n}
=
\sum_c
S^A_{0,m,c}
S^B_{0,n,c}
\left[
\sum_{k \in c}
Q^A_{m,k}
Q^B_{n,k}
S^A_{1,m,g}
S^B_{1,n,g}
\right]
}
\]

Define:

\[
P_c[m,n]
=
\sum_{k \in c}
Q^A_{m,k}
Q^B_{n,k}
S^A_{1,m,g}
S^B_{1,n,g}
\]

Then:

\[
\boxed{
Y_{m,n}
=
\sum_c
P_c[m,n]
S^A_{0,m,c}
S^B_{0,n,c}
}
\]

This is the mathematical essence of DualScale GEMM.

---

## 8. Why the Coarse Scale Helps

The main benefit is improved scale resolution.

### Standard MXFP4

\[
s_{\text{effective}}
=
2^e
\]

### DualScale MXFP4

\[
s_{\text{effective}}
=
s_{\text{coarse}}^{FP32}
\cdot
2^e
\]

Therefore DualScale keeps the efficient MXFP4 fine-scale representation while introducing an arbitrary floating-point offset for the scale grid.

This can reduce:

- clipping;
- underflow;
- reconstruction error;
- activation error;
- GEMM output error.

The benefit tends to be more important for FP4 than for higher-precision formats because FP4 has very limited representational capacity.

---

## 9. What DualScale Does Not Do

DualScale does **not** give every 32-element group an independent arbitrary FP32 scale.

Within one 512-element block:

\[
s_g
=
s_c^{\text{coarse}}
\cdot
2^{e_g}
\]

So all 16 fine groups still share the same FP32 base factor.

DualScale therefore provides a compromise:

```text
Per-32 arbitrary FP32 scale
    -> highest flexibility
    -> higher scale storage / complexity

DualScale:
    FP32 /512 × E8M0 /32
    -> much better scale flexibility
    -> low metadata overhead

Standard MXFP4:
    E8M0 /32 only
    -> simplest
    -> most restrictive scale grid
```

---

## 10. Scale Storage Overhead

For every 512 values:

### Standard MXFP4

There are:

\[
512/32 = 16
\]

fine scales.

With one byte per E8M0 scale:

\[
16\text{ bytes}
\]

### DualScale

Add one FP32 coarse scale:

\[
4\text{ bytes}
\]

So scale metadata increases approximately from:

\[
16 \rightarrow 20 \text{ bytes}
\]

per 512 values.

The additional metadata is small compared with the FP4 payload:

\[
512 \times 4\text{ bits}
=
256\text{ bytes}
\]

Thus the extra coarse-scale storage is only:

\[
\frac{4}{256}\approx1.56\%
\]

of the FP4 payload size.

---

## 11. Summary

Standard MXFP4 represents values as:

\[
\boxed{
x \approx FP4 \times E8M0_{/32}
}
\]

DualScale extends this to:

\[
\boxed{
x
\approx
FP4
\times
E8M0_{/32}
\times
FP32_{/512}
}
\]

The important difference is that E8M0 alone only provides power-of-two scales, while the FP32 coarse scale shifts that power-of-two grid to better match the local tensor distribution.

The effective local scale becomes:

\[
\boxed{
S_{\text{effective}}
=
S_{\text{coarse}}^{FP32}
\times
S_{\text{fine}}^{E8M0}
}
\]

For GEMM:

\[
\boxed{
Y_{m,n}
=
\sum_c
S^A_{0,m,c}
S^B_{0,n,c}
P_c[m,n]
}
\]

where \(P_c\) is the standard fine-scaled MXFP4 contribution from one 512-element K block.

The core idea is therefore:

> **Use an FP32 scale at a coarse granularity to improve scale resolution, while retaining standard per-32 E8M0 microscaling and FP4 data representation.**
