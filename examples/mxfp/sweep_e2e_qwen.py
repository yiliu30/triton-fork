"""
End-to-end sweep: Compare all MXFP attention variants on Qwen3-0.6B generation.

Tests 5 variants:
1. FP16 SDPA (baseline)
2. Pure MXFP8
3. Pure MXFP4
4. MXFP8 QK + MXFP4 PV
5. MXFP4 QK + MXFP8 PV

Metrics: perplexity, token match rate, logit cosine similarity, generation quality.

Usage:
    export C_INCLUDE_PATH=$HOME/.local/include:$HOME/.local/include/python3.12
    CUDA_VISIBLE_DEVICES=3 /home/yiliu7/workspace/venvs/omni/bin/python sweep_e2e_qwen.py
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/yiliu7/workspace/triton-fork/examples/mxfp")

from mxfp4_flash_attention import quantize_to_mxfp4, mxfp4_flash_attention
from mxfp8_flash_attention import quantize_to_mxfp8, mxfp8_flash_attention
from mxfp8_qk_mxfp4_pv_flash_attention import mixed_mxfp8qk_mxfp4pv_flash_attention
from mxfp4_qk_mxfp8_pv_flash_attention import mixed_mxfp4qk_mxfp8pv_flash_attention

DEVICE = "cuda:0"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# ============================================================================
# SDPA replacements for each variant
# ============================================================================

BLOCK_M, BLOCK_N = 128, 64


def _pad_inputs(query_states, key_states, value_states, is_causal):
    """Pad M and N to required block sizes."""
    B, H_q, M, D = query_states.shape
    H_kv = key_states.shape[1]
    N = key_states.shape[2]

    if H_kv != H_q:
        n_rep = H_q // H_kv
        key_states = key_states.repeat_interleave(n_rep, dim=1)
        value_states = value_states.repeat_interleave(n_rep, dim=1)

    pad_m = (BLOCK_M - M % BLOCK_M) % BLOCK_M
    pad_n = (BLOCK_N - N % BLOCK_N) % BLOCK_N
    M_padded = M + pad_m
    N_target = M_padded if is_causal else N + pad_n
    N_target = N_target + (BLOCK_N - N_target % BLOCK_N) % BLOCK_N
    pad_n = N_target - N

    if pad_m > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_m))
    if pad_n > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_n))
        value_states = F.pad(value_states, (0, 0, 0, pad_n))

    return query_states, key_states, value_states, M


def _make_sdpa_replacement(variant):
    """Create an SDPA replacement function for the given variant."""

    def replacement(query_states, key_states, value_states, attn_mask=None,
                    dropout_p=0.0, scale=None, is_causal=False, **kwargs):
        B, H_q, M, D = query_states.shape

        if D not in (64, 128):
            return F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attn_mask, dropout_p=dropout_p,
                scale=scale, is_causal=is_causal)

        q_pad, k_pad, v_pad, M_orig = _pad_inputs(
            query_states, key_states, value_states, is_causal)

        H = q_pad.shape[1]
        M_padded = q_pad.shape[2]
        N_padded = k_pad.shape[2]
        causal = is_causal and (M_padded == N_padded)
        sm_scale = scale if scale is not None else (1.0 / (D ** 0.5))

        if variant == "mxfp4":
            q_packed, q_scale = quantize_to_mxfp4(q_pad.float())
            k_packed, k_scale = quantize_to_mxfp4(k_pad.float())
            v_t = v_pad.float().permute(0, 1, 3, 2).contiguous()
            v_flat = v_t.reshape(-1, N_padded)
            v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
            v_packed = v_packed_flat.reshape(B, H, D, N_padded // 2)
            v_scale = v_scale_flat.reshape(B, H, D, N_padded // 32)
            out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                         q_scale, k_scale, v_scale,
                                         causal=causal, sm_scale=sm_scale)

        elif variant == "mxfp8":
            q_fp8, q_scale = quantize_to_mxfp8(q_pad.float())
            k_fp8, k_scale = quantize_to_mxfp8(k_pad.float())
            v_t = v_pad.float().permute(0, 1, 3, 2).contiguous()
            v_flat_2d = v_t.reshape(-1, N_padded)
            v_fp8_flat, v_scale_flat = quantize_to_mxfp8(v_flat_2d)
            v_fp8 = v_fp8_flat.reshape(B, H, D, N_padded).permute(0, 1, 3, 2).contiguous()
            v_scale = v_scale_flat.reshape(B, H, D, N_padded // 32)
            out = mxfp8_flash_attention(q_fp8, k_fp8, v_fp8,
                                         q_scale, k_scale, v_scale,
                                         causal=causal, sm_scale=sm_scale)

        elif variant == "mxfp8qk_mxfp4pv":
            q_fp8, q_scale = quantize_to_mxfp8(q_pad.float())
            k_fp8, k_scale = quantize_to_mxfp8(k_pad.float())
            v_t = v_pad.float().permute(0, 1, 3, 2).contiguous()
            v_flat = v_t.reshape(-1, N_padded)
            v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
            v_packed = v_packed_flat.reshape(B, H, D, N_padded // 2)
            v_scale = v_scale_flat.reshape(B, H, D, N_padded // 32)
            out = mixed_mxfp8qk_mxfp4pv_flash_attention(
                q_fp8, k_fp8, v_packed, q_scale, k_scale, v_scale,
                causal=causal, sm_scale=sm_scale)

        elif variant == "mxfp4qk_mxfp8pv":
            q_packed, q_scale = quantize_to_mxfp4(q_pad.float())
            k_packed, k_scale = quantize_to_mxfp4(k_pad.float())
            v_t = v_pad.float().permute(0, 1, 3, 2).contiguous()
            v_flat_2d = v_t.reshape(-1, N_padded)
            v_fp8_flat, v_scale_flat = quantize_to_mxfp8(v_flat_2d)
            v_fp8 = v_fp8_flat.reshape(B, H, D, N_padded).permute(0, 1, 3, 2).contiguous()
            v_scale = v_scale_flat.reshape(B, H, D, N_padded // 32)
            out = mixed_mxfp4qk_mxfp8pv_flash_attention(
                q_packed, k_packed, v_fp8, q_scale, k_scale, v_scale,
                causal=causal, sm_scale=sm_scale)

        out = out[:, :, :M_orig, :].to(query_states.dtype)
        return out

    return replacement


# ============================================================================
# Model utilities
# ============================================================================

_original_sdpa = F.scaled_dot_product_attention


def patch_model(variant):
    """Patch SDPA with the given variant. Use variant=None to restore."""
    if variant is None:
        torch.nn.functional.scaled_dot_product_attention = _original_sdpa
    else:
        torch.nn.functional.scaled_dot_product_attention = _make_sdpa_replacement(variant)


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=64):
    """Generate text with greedy decoding. Returns text + last-token logits."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    outputs = model(input_ids)
    prompt_logits = outputs.logits[0, -1, :]  # last token logits

    gen_output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    generated_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
    return generated_text, prompt_logits


@torch.no_grad()
def compute_perplexity(model, tokenizer, text):
    """Compute perplexity on a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    outputs = model(input_ids, labels=input_ids)
    return torch.exp(outputs.loss).item()


# ============================================================================
# Main sweep
# ============================================================================

VARIANTS = [
    ("fp16", "FP16 SDPA (baseline)"),
    ("mxfp8", "Pure MXFP8"),
    ("mxfp4", "Pure MXFP4"),
    ("mxfp8qk_mxfp4pv", "MXFP8 QK + MXFP4 PV"),
    ("mxfp4qk_mxfp8pv", "MXFP4 QK + MXFP8 PV"),
]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"",
    "In machine learning, backpropagation is",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    "The theory of relativity states that",
]

EVAL_TEXTS = [
    "The transformer architecture consists of an encoder and a decoder. The encoder processes the input sequence and produces a set of representations. The decoder then generates the output sequence one token at a time, attending to both the encoder output and previously generated tokens.",
    "Python is a high-level programming language known for its simple syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "Large language models are trained on massive datasets of text from the internet. They learn statistical patterns in language that allow them to generate coherent text, answer questions, and perform various natural language tasks.",
]


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Model: {model.config.num_attention_heads} heads, head_dim={head_dim}, "
          f"kv_heads={model.config.num_key_value_heads}")
    print()

    # Storage for results
    results = {}

    for variant_key, variant_name in VARIANTS:
        print("=" * 70)
        print(f"  {variant_name}")
        print("=" * 70)

        # Patch
        if variant_key == "fp16":
            patch_model(None)
        else:
            patch_model(variant_key)

        # Generation
        gen_outputs = []
        logits_list = []
        for prompt in PROMPTS:
            text, logits = generate_text(model, tokenizer, prompt)
            gen_outputs.append(text)
            logits_list.append(logits)
            out_snippet = repr(text[len(prompt):][:80])
            print(f"  [{prompt[:40]:<40}] -> {out_snippet}")

        # Perplexity
        ppls = []
        for text in EVAL_TEXTS:
            ppl = compute_perplexity(model, tokenizer, text)
            ppls.append(ppl)

        print(f"  Perplexities: {[f'{p:.2f}' for p in ppls]}")
        print()

        results[variant_key] = {
            "name": variant_name,
            "gen_outputs": gen_outputs,
            "logits": logits_list,
            "ppls": ppls,
        }

    # Restore
    patch_model(None)

    # ========================================================================
    # Summary comparison
    # ========================================================================
    print()
    print("=" * 90)
    print("  SUMMARY: End-to-End Comparison on Qwen3-0.6B")
    print("=" * 90)

    # --- Perplexity table ---
    print("\n  Perplexity (lower is better):")
    print(f"  {'Variant':<30} | {'Text 0':<10} | {'Text 1':<10} | {'Text 2':<10} | {'Avg':<10} | {'Delta':<10}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    baseline_avg_ppl = sum(results["fp16"]["ppls"]) / len(results["fp16"]["ppls"])

    for variant_key, variant_name in VARIANTS:
        ppls = results[variant_key]["ppls"]
        avg_ppl = sum(ppls) / len(ppls)
        delta = avg_ppl - baseline_avg_ppl
        marker = "" if variant_key == "fp16" else f"{delta:+.2f}"
        print(f"  {variant_name:<30} | {ppls[0]:<10.2f} | {ppls[1]:<10.2f} | {ppls[2]:<10.2f} | "
              f"{avg_ppl:<10.2f} | {marker:<10}")

    # --- Logit cosine similarity ---
    print(f"\n  Logit Cosine Similarity vs FP16 (per prompt, higher is better):")
    print(f"  {'Variant':<30} | ", end="")
    for i in range(len(PROMPTS)):
        print(f"{'P'+str(i):<10} | ", end="")
    print(f"{'Avg':<10}")
    print(f"  {'-'*30}-+-" + "-+-".join(["-"*10]*len(PROMPTS)) + f"-+-{'-'*10}")

    fp16_logits = results["fp16"]["logits"]

    for variant_key, variant_name in VARIANTS:
        if variant_key == "fp16":
            continue
        cos_sims = []
        for i in range(len(PROMPTS)):
            cs = F.cosine_similarity(
                fp16_logits[i].float().unsqueeze(0),
                results[variant_key]["logits"][i].float().unsqueeze(0)
            ).item()
            cos_sims.append(cs)
        avg_cs = sum(cos_sims) / len(cos_sims)
        print(f"  {variant_name:<30} | ", end="")
        for cs in cos_sims:
            print(f"{cs:<10.6f} | ", end="")
        print(f"{avg_cs:<10.6f}")

    # --- Token match rate ---
    print(f"\n  Token Match Rate vs FP16 (greedy generation, higher is better):")
    print(f"  {'Variant':<30} | ", end="")
    for i in range(len(PROMPTS)):
        print(f"{'P'+str(i):<10} | ", end="")
    print(f"{'Total':<10}")
    print(f"  {'-'*30}-+-" + "-+-".join(["-"*10]*len(PROMPTS)) + f"-+-{'-'*10}")

    fp16_gens = results["fp16"]["gen_outputs"]

    for variant_key, variant_name in VARIANTS:
        if variant_key == "fp16":
            continue
        total_match = 0
        total_tokens = 0
        per_prompt_pct = []
        for i in range(len(PROMPTS)):
            orig_tokens = tokenizer.encode(fp16_gens[i])
            var_tokens = tokenizer.encode(results[variant_key]["gen_outputs"][i])
            min_len = min(len(orig_tokens), len(var_tokens))
            matches = sum(1 for a, b in zip(orig_tokens[:min_len], var_tokens[:min_len]) if a == b)
            total_match += matches
            total_tokens += min_len
            pct = matches / min_len * 100 if min_len > 0 else 0
            per_prompt_pct.append(pct)

        overall_pct = total_match / total_tokens * 100 if total_tokens > 0 else 0
        print(f"  {variant_name:<30} | ", end="")
        for pct in per_prompt_pct:
            print(f"{pct:<10.1f}| ", end="")
        print(f"{overall_pct:<10.1f}%")

    # --- Generation examples (show divergences) ---
    print(f"\n  Generation Examples (showing first divergence from FP16):")
    print(f"  {'-'*85}")
    for i, prompt in enumerate(PROMPTS[:3]):  # Show first 3
        print(f"\n  Prompt: {prompt!r}")
        fp16_out = fp16_gens[i][len(prompt):][:100]
        print(f"    FP16:              {repr(fp16_out)}")
        for variant_key, variant_name in VARIANTS[1:]:
            var_out = results[variant_key]["gen_outputs"][i][len(prompt):][:100]
            if var_out == fp16_out:
                print(f"    {variant_name:<22} [IDENTICAL]")
            else:
                print(f"    {variant_name:<22} {repr(var_out)}")

    print()
    print("=" * 90)
    print("  RECOMMENDATION")
    print("=" * 90)
    print()

    # Determine best mixed variant
    mxfp8qk_avg_ppl = sum(results["mxfp8qk_mxfp4pv"]["ppls"]) / len(results["mxfp8qk_mxfp4pv"]["ppls"])
    mxfp4qk_avg_ppl = sum(results["mxfp4qk_mxfp8pv"]["ppls"]) / len(results["mxfp4qk_mxfp8pv"]["ppls"])

    if mxfp4qk_avg_ppl < mxfp8qk_avg_ppl:
        print("  MXFP4 QK + MXFP8 PV wins: lower perplexity, preserving V precision matters more.")
        print(f"  PPL: {mxfp4qk_avg_ppl:.2f} vs {mxfp8qk_avg_ppl:.2f} (delta = {mxfp4qk_avg_ppl - mxfp8qk_avg_ppl:.2f})")
    else:
        print("  MXFP8 QK + MXFP4 PV wins: lower perplexity, preserving QK precision matters more.")
        print(f"  PPL: {mxfp8qk_avg_ppl:.2f} vs {mxfp4qk_avg_ppl:.2f} (delta = {mxfp8qk_avg_ppl - mxfp4qk_avg_ppl:.2f})")

    print()


if __name__ == "__main__":
    main()
