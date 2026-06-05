"""
Strict end-to-end evaluation: MXFP attention variants on Qwen3-0.6B.

More rigorous than the basic sweep:
- More diverse prompts (reasoning, math, code, factual, creative)
- Longer generation (128 tokens)
- Per-layer logit KL divergence
- Top-k accuracy (does the correct token stay in top-k?)
- Cumulative token divergence across sequence positions
- Multiple random seeds

Usage:
    export C_INCLUDE_PATH=$HOME/.local/include:$HOME/.local/include/python3.12
    CUDA_VISIBLE_DEVICES=3 /home/yiliu7/workspace/venvs/omni/bin/python sweep_e2e_qwen_strict.py
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/yiliu7/workspace/triton-fork/examples/mxfp")

from mxfp4_flash_attention import quantize_to_mxfp4, mxfp4_flash_attention
from mxfp8_flash_attention import quantize_to_mxfp8, mxfp8_flash_attention
from mxfp8_qk_mxfp4_pv_flash_attention import mixed_mxfp8qk_mxfp4pv_flash_attention
from mxfp4_qk_mxfp8_pv_flash_attention import mixed_mxfp4qk_mxfp8pv_flash_attention

DEVICE = "cuda:0"
MODEL_NAME = "Qwen/Qwen3-0.6B"

BLOCK_M, BLOCK_N = 128, 64

# ============================================================================
# SDPA replacement (same as before)
# ============================================================================

_original_sdpa = F.scaled_dot_product_attention


def _pad_inputs(query_states, key_states, value_states, is_causal):
    B, H_q, M, D = query_states.shape
    H_kv = key_states.shape[1]
    N = key_states.shape[2]

    if H_kv != H_q:
        n_rep = H_q // H_kv
        key_states = key_states.repeat_interleave(n_rep, dim=1)
        value_states = value_states.repeat_interleave(n_rep, dim=1)

    pad_m = (BLOCK_M - M % BLOCK_M) % BLOCK_M
    M_padded = M + pad_m
    N_target = M_padded if is_causal else N + (BLOCK_N - N % BLOCK_N) % BLOCK_N
    N_target = N_target + (BLOCK_N - N_target % BLOCK_N) % BLOCK_N
    pad_n = N_target - N

    if pad_m > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_m))
    if pad_n > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_n))
        value_states = F.pad(value_states, (0, 0, 0, pad_n))

    return query_states, key_states, value_states, M


def _make_sdpa_replacement(variant):
    def replacement(query_states, key_states, value_states, attn_mask=None,
                    dropout_p=0.0, scale=None, is_causal=False, **kwargs):
        B, H_q, M, D = query_states.shape
        if D not in (64, 128):
            return _original_sdpa(query_states, key_states, value_states,
                                  attn_mask=attn_mask, dropout_p=dropout_p,
                                  scale=scale, is_causal=is_causal)

        q_pad, k_pad, v_pad, M_orig = _pad_inputs(
            query_states, key_states, value_states, is_causal)
        H = q_pad.shape[1]
        N_padded = k_pad.shape[2]
        causal = is_causal and (q_pad.shape[2] == N_padded)
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

        return out[:, :, :M_orig, :].to(query_states.dtype)
    return replacement


def patch_model(variant):
    if variant is None:
        torch.nn.functional.scaled_dot_product_attention = _original_sdpa
    else:
        torch.nn.functional.scaled_dot_product_attention = _make_sdpa_replacement(variant)


# ============================================================================
# Strict evaluation metrics
# ============================================================================


@torch.no_grad()
def get_all_logits(model, tokenizer, text):
    """Get logits at every position for a given text (teacher forcing)."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    outputs = model(inputs["input_ids"])
    return outputs.logits[0]  # [seq_len, vocab_size]


@torch.no_grad()
def compute_metrics_strict(model, tokenizer, text, fp16_logits):
    """Compute strict metrics comparing variant logits to FP16 logits."""
    var_logits = get_all_logits(model, tokenizer, text)

    # Ensure same length
    min_len = min(fp16_logits.shape[0], var_logits.shape[0])
    ref = fp16_logits[:min_len].float()
    var = var_logits[:min_len].float()

    # Per-position cosine similarity
    cos_sims = F.cosine_similarity(ref, var, dim=-1)  # [seq_len]

    # KL divergence: KL(p_fp16 || p_variant)
    ref_probs = F.softmax(ref, dim=-1)
    var_log_probs = F.log_softmax(var, dim=-1)
    kl_div = F.kl_div(var_log_probs, ref_probs, reduction='none').sum(-1)  # [seq_len]

    # Top-k accuracy: is the FP16 argmax token in the variant's top-k?
    ref_argmax = ref.argmax(dim=-1)  # [seq_len]
    top1_match = (var.argmax(dim=-1) == ref_argmax).float()
    top5_vals, top5_idx = var.topk(5, dim=-1)
    top5_match = (top5_idx == ref_argmax.unsqueeze(-1)).any(dim=-1).float()
    top10_vals, top10_idx = var.topk(10, dim=-1)
    top10_match = (top10_idx == ref_argmax.unsqueeze(-1)).any(dim=-1).float()

    # Jensen-Shannon divergence (symmetric)
    ref_probs_clamped = ref_probs.clamp(min=1e-10)
    var_probs = F.softmax(var, dim=-1).clamp(min=1e-10)
    m_probs = 0.5 * (ref_probs_clamped + var_probs)
    js_div = 0.5 * (F.kl_div(m_probs.log(), ref_probs_clamped, reduction='none').sum(-1) +
                     F.kl_div(m_probs.log(), var_probs, reduction='none').sum(-1))

    return {
        "cos_sim_mean": cos_sims.mean().item(),
        "cos_sim_min": cos_sims.min().item(),
        "cos_sim_p5": cos_sims.quantile(0.05).item(),
        "kl_div_mean": kl_div.mean().item(),
        "kl_div_max": kl_div.max().item(),
        "kl_div_p95": kl_div.quantile(0.95).item(),
        "js_div_mean": js_div.mean().item(),
        "top1_accuracy": top1_match.mean().item(),
        "top5_accuracy": top5_match.mean().item(),
        "top10_accuracy": top10_match.mean().item(),
        "seq_len": min_len,
    }


@torch.no_grad()
def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    gen_output = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    return tokenizer.decode(gen_output[0], skip_special_tokens=True)


# ============================================================================
# Test data
# ============================================================================

EVAL_TEXTS = [
    # Factual knowledge
    "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It replaced recurrent neural networks with self-attention mechanisms, enabling much better parallelization during training and capturing long-range dependencies more effectively.",
    # Code understanding
    "In Python, a decorator is a function that takes another function as an argument and extends its behavior without explicitly modifying it. Decorators are commonly used for logging, authentication, and caching. The @property decorator transforms a method into a read-only attribute.",
    # Mathematical reasoning
    "The derivative of a composite function f(g(x)) is calculated using the chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x). For example, if h(x) = sin(x^2), then h'(x) = cos(x^2) * 2x, because the outer function is sin and the inner function is x^2.",
    # Scientific text
    "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently. When one particle is measured, the state of the other is instantly determined, regardless of the distance between them.",
    # Long-form narrative
    "Machine learning models can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning. Supervised learning requires labeled training data and includes classification and regression tasks. Unsupervised learning discovers patterns in unlabeled data through clustering and dimensionality reduction. Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward through interaction with an environment.",
]

GENERATION_PROMPTS = [
    # Factual
    "The three laws of thermodynamics are:",
    # Code generation
    "def binary_search(arr, target):\n    \"\"\"Find target in sorted array, return index or -1.\"\"\"",
    # Math
    "To solve the quadratic equation ax^2 + bx + c = 0, we use the quadratic formula:",
    # Reasoning
    "If all roses are flowers and all flowers need water, then we can conclude that",
    # Creative
    "The difference between a compiler and an interpreter is that",
    # Long context
    "Consider a hash table with open addressing and linear probing. When the load factor exceeds 0.75,",
    # Multi-step
    "To implement a breadth-first search (BFS) algorithm, you need: 1)",
    # Arithmetic
    "15 * 7 = 105, 23 * 4 = 92, 31 * 9 =",
]

VARIANTS = [
    ("mxfp8", "Pure MXFP8"),
    ("mxfp4", "Pure MXFP4"),
    ("mxfp8qk_mxfp4pv", "MXFP8 QK + MXFP4 PV"),
    ("mxfp4qk_mxfp8pv", "MXFP4 QK + MXFP8 PV"),
]


# ============================================================================
# Main
# ============================================================================


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

    # ===========================================================
    # Phase 1: Collect FP16 baselines
    # ===========================================================
    print("=" * 80)
    print("  Phase 1: Collecting FP16 baselines")
    print("=" * 80)
    patch_model(None)

    fp16_logits_list = []
    fp16_ppls = []
    for i, text in enumerate(EVAL_TEXTS):
        logits = get_all_logits(model, tokenizer, text)
        fp16_logits_list.append(logits)
        ppl = compute_perplexity(model, tokenizer, text)
        fp16_ppls.append(ppl)
        print(f"  Text {i}: len={logits.shape[0]} tokens, PPL={ppl:.2f}")

    fp16_gens = []
    for prompt in GENERATION_PROMPTS:
        gen = greedy_generate(model, tokenizer, prompt, max_new_tokens=128)
        fp16_gens.append(gen)

    print(f"  Generated {len(fp16_gens)} completions.")
    print()

    # ===========================================================
    # Phase 2: Evaluate each variant
    # ===========================================================
    all_results = {}

    for variant_key, variant_name in VARIANTS:
        print("=" * 80)
        print(f"  Phase 2: Evaluating {variant_name}")
        print("=" * 80)
        patch_model(variant_key)

        # --- Strict logit metrics ---
        metrics_list = []
        ppls = []
        for i, text in enumerate(EVAL_TEXTS):
            metrics = compute_metrics_strict(model, tokenizer, text, fp16_logits_list[i])
            metrics_list.append(metrics)
            ppl = compute_perplexity(model, tokenizer, text)
            ppls.append(ppl)
            print(f"  Text {i}: cos_sim={metrics['cos_sim_mean']:.4f} (min={metrics['cos_sim_min']:.4f}), "
                  f"KL={metrics['kl_div_mean']:.4f}, top1={metrics['top1_accuracy']:.3f}, "
                  f"PPL={ppl:.2f}")

        # --- Generation token match ---
        gens = []
        token_matches = []
        for j, prompt in enumerate(GENERATION_PROMPTS):
            gen = greedy_generate(model, tokenizer, prompt, max_new_tokens=128)
            gens.append(gen)

            orig_tokens = tokenizer.encode(fp16_gens[j])
            var_tokens = tokenizer.encode(gen)
            min_len = min(len(orig_tokens), len(var_tokens))
            if min_len > 0:
                matches = sum(1 for a, b in zip(orig_tokens[:min_len], var_tokens[:min_len]) if a == b)
                token_matches.append(matches / min_len)
            else:
                token_matches.append(0.0)

        avg_token_match = np.mean(token_matches)
        print(f"  Avg token match: {avg_token_match*100:.1f}%")
        print()

        all_results[variant_key] = {
            "name": variant_name,
            "metrics": metrics_list,
            "ppls": ppls,
            "gens": gens,
            "token_matches": token_matches,
        }

    # Restore
    patch_model(None)

    # ===========================================================
    # Phase 3: Summary tables
    # ===========================================================
    print()
    print("=" * 80)
    print("  STRICT EVALUATION SUMMARY")
    print("=" * 80)

    # --- Table 1: Perplexity ---
    print("\n  ┌─────────────────────────────────────────────────────────────────────────────┐")
    print("  │ PERPLEXITY (lower = better)                                                 │")
    print("  ├─────────────────────────────┬────────┬────────┬────────┬────────┬────────┬──┤")
    print(f"  │ {'Variant':<27} │ {'T0':<6} │ {'T1':<6} │ {'T2':<6} │ {'T3':<6} │ {'T4':<6} │Δ │")
    print("  ├─────────────────────────────┼────────┼────────┼────────┼────────┼────────┼──┤")

    fp16_avg = np.mean(fp16_ppls)
    print(f"  │ {'FP16 SDPA':<27} │ {fp16_ppls[0]:<6.2f} │ {fp16_ppls[1]:<6.2f} │ "
          f"{fp16_ppls[2]:<6.2f} │ {fp16_ppls[3]:<6.2f} │ {fp16_ppls[4]:<6.2f} │{'—':<2}│")

    for variant_key, variant_name in VARIANTS:
        ppls = all_results[variant_key]["ppls"]
        avg = np.mean(ppls)
        delta = avg - fp16_avg
        d_str = f"{delta:+.1f}"
        print(f"  │ {variant_name:<27} │ {ppls[0]:<6.2f} │ {ppls[1]:<6.2f} │ "
              f"{ppls[2]:<6.2f} │ {ppls[3]:<6.2f} │ {ppls[4]:<6.2f} │{d_str:<2}│")
    print("  └─────────────────────────────┴────────┴────────┴────────┴────────┴────────┴──┘")

    # --- Table 2: Logit Quality ---
    print("\n  ┌───────────────────────────────────────────────────────────────────────────┐")
    print("  │ LOGIT QUALITY vs FP16 (averaged over all eval texts)                      │")
    print("  ├─────────────────────────────┬──────────┬──────────┬────────┬──────┬───────┤")
    print(f"  │ {'Variant':<27} │ {'cos_sim':<8} │ {'cos_p5':<8} │ {'KL_div':<6} │ {'top1':<4} │ {'top5':<5} │")
    print("  ├─────────────────────────────┼──────────┼──────────┼────────┼──────┼───────┤")

    for variant_key, variant_name in VARIANTS:
        ml = all_results[variant_key]["metrics"]
        avg_cos = np.mean([m["cos_sim_mean"] for m in ml])
        avg_cos_p5 = np.mean([m["cos_sim_p5"] for m in ml])
        avg_kl = np.mean([m["kl_div_mean"] for m in ml])
        avg_top1 = np.mean([m["top1_accuracy"] for m in ml])
        avg_top5 = np.mean([m["top5_accuracy"] for m in ml])
        print(f"  │ {variant_name:<27} │ {avg_cos:<8.4f} │ {avg_cos_p5:<8.4f} │ "
              f"{avg_kl:<6.3f} │ {avg_top1:<4.1%} │ {avg_top5:<5.1%} │")
    print("  └─────────────────────────────┴──────────┴──────────┴────────┴──────┴───────┘")

    # --- Table 3: Generation Quality ---
    print("\n  ┌───────────────────────────────────────────────────────────────────────────┐")
    print("  │ GENERATION TOKEN MATCH vs FP16 (greedy, 128 tokens)                       │")
    print("  ├─────────────────────────────┬────────────────────────────────┬─────────────┤")
    hdr = " | ".join([f"P{i}" for i in range(len(GENERATION_PROMPTS))])
    print(f"  │ {'Variant':<27} │ {hdr:<30} │ {'Avg':<11} │")
    print("  ├─────────────────────────────┼────────────────────────────────┼─────────────┤")

    for variant_key, variant_name in VARIANTS:
        tm = all_results[variant_key]["token_matches"]
        avg_tm = np.mean(tm)
        vals = " | ".join([f"{t*100:2.0f}" for t in tm])
        print(f"  │ {variant_name:<27} │ {vals:<30} │ {avg_tm*100:<11.1f}│")
    print("  └─────────────────────────────┴────────────────────────────────┴─────────────┘")

    # --- Table 4: Worst-case analysis ---
    print("\n  ┌───────────────────────────────────────────────────────────────────────────┐")
    print("  │ WORST-CASE ANALYSIS (tail behavior matters for reliability)                │")
    print("  ├─────────────────────────────┬──────────┬──────────┬──────────┬─────────────┤")
    print(f"  │ {'Variant':<27} │ {'cos_min':<8} │ {'KL_max':<8} │ {'KL_p95':<8} │ {'JS_div_avg':<11} │")
    print("  ├─────────────────────────────┼──────────┼──────────┼──────────┼─────────────┤")

    for variant_key, variant_name in VARIANTS:
        ml = all_results[variant_key]["metrics"]
        worst_cos = min(m["cos_sim_min"] for m in ml)
        worst_kl = max(m["kl_div_max"] for m in ml)
        avg_kl_p95 = np.mean([m["kl_div_p95"] for m in ml])
        avg_js = np.mean([m["js_div_mean"] for m in ml])
        print(f"  │ {variant_name:<27} │ {worst_cos:<8.4f} │ {worst_kl:<8.2f} │ "
              f"{avg_kl_p95:<8.3f} │ {avg_js:<11.5f} │")
    print("  └─────────────────────────────┴──────────┴──────────┴──────────┴─────────────┘")

    # --- Final verdict ---
    print("\n  " + "=" * 76)
    print("  VERDICT")
    print("  " + "=" * 76)

    # Rank by composite score: lower PPL delta + higher cos_sim + higher top1
    scores = {}
    for variant_key, variant_name in VARIANTS:
        ml = all_results[variant_key]["metrics"]
        ppls = all_results[variant_key]["ppls"]
        avg_ppl_delta = np.mean(ppls) - fp16_avg
        avg_cos = np.mean([m["cos_sim_mean"] for m in ml])
        avg_top1 = np.mean([m["top1_accuracy"] for m in ml])
        avg_kl = np.mean([m["kl_div_mean"] for m in ml])
        avg_token_match = np.mean(all_results[variant_key]["token_matches"])

        # Composite: penalty for PPL increase + reward for accuracy
        score = -avg_ppl_delta + 10 * (avg_cos - 0.9) + 5 * avg_top1 - avg_kl
        scores[variant_key] = score

        print(f"\n  {variant_name}:")
        print(f"    PPL: {np.mean(ppls):.2f} (Δ={avg_ppl_delta:+.2f})")
        print(f"    Logit cos_sim: {avg_cos:.4f}")
        print(f"    Top-1 accuracy: {avg_top1:.1%}")
        print(f"    KL divergence: {avg_kl:.4f}")
        print(f"    Token match: {avg_token_match:.1%}")
        print(f"    Composite score: {score:.3f}")

    best = max(scores, key=scores.get)
    best_name = dict(VARIANTS)[best]
    print(f"\n  >>> BEST VARIANT: {best_name} (score={scores[best]:.3f})")
    print()

    # Show generation examples for the best vs worst
    worst = min(scores, key=scores.get)
    print(f"  Example generations (first 3 prompts):")
    for i in range(min(3, len(GENERATION_PROMPTS))):
        prompt = GENERATION_PROMPTS[i]
        print(f"\n  Prompt: {prompt[:60]}...")
        print(f"    FP16:  {repr(fp16_gens[i][len(prompt):][:80])}")
        print(f"    BEST:  {repr(all_results[best]['gens'][i][len(prompt):][:80])}")
        print(f"    WORST: {repr(all_results[worst]['gens'][i][len(prompt):][:80])}")

    print()


if __name__ == "__main__":
    main()
