"""
End-to-end accuracy test: Replace SDPA with MXFP4 Flash Attention in Qwen3-0.6B.

Compares generation quality between:
1. Original model (FP16 SDPA)
2. Model with MXFP4 attention (Q/K/V quantized to E2M1 on-the-fly per layer)

Usage:
    source /var/tmp/test-mma/export_local_uv_env.sh
    CUDA_VISIBLE_DEVICES=0 python e2e_qwen_mxfp4.py
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/software/hshen/yiliu7/triton-fork/examples/mxfp")
from mxfp4_flash_attention import quantize_to_mxfp4, dequantize_mxfp4, mxfp4_flash_attention

DEVICE = "cuda:0"
MODEL_NAME = "Qwen/Qwen3-0.6B"


def mxfp4_sdpa_replacement(query_states, key_states, value_states, attn_mask=None,
                            dropout_p=0.0, scale=None, is_causal=False, **kwargs):
    """
    Drop-in replacement for scaled_dot_product_attention using MXFP4 kernel.
    Quantizes Q/K/V to E2M1 on-the-fly, runs our kernel, returns FP16 output.
    """
    B, H_q, M, D = query_states.shape
    H_kv = key_states.shape[1]
    N = key_states.shape[2]

    # Ensure head_dim is supported (64 or 128)
    if D not in (64, 128):
        return F.scaled_dot_product_attention(query_states, key_states, value_states,
                                              attn_mask=attn_mask, dropout_p=dropout_p,
                                              scale=scale, is_causal=is_causal)

    # Handle GQA: expand K/V heads to match Q heads
    if H_kv != H_q:
        n_rep = H_q // H_kv
        key_states = key_states.repeat_interleave(n_rep, dim=1)
        value_states = value_states.repeat_interleave(n_rep, dim=1)
    H = H_q

    # Pad M and N to multiples of BLOCK_M and BLOCK_N
    BLOCK_M, BLOCK_N = 128, 64
    pad_m = (BLOCK_M - M % BLOCK_M) % BLOCK_M
    # N must also be multiple of BLOCK_N
    pad_n = (BLOCK_N - N % BLOCK_N) % BLOCK_N
    # For causal: N must equal M_padded for correct masking
    M_padded = M + pad_m
    N_target = M_padded if is_causal else N + pad_n
    # Ensure N_target is multiple of BLOCK_N
    N_target = N_target + (BLOCK_N - N_target % BLOCK_N) % BLOCK_N
    pad_n = N_target - N

    if pad_m > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_m))
    if pad_n > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_n))
        value_states = F.pad(value_states, (0, 0, 0, pad_n))

    M_padded = query_states.shape[2]
    N_padded = key_states.shape[2]

    # Quantize Q, K to MXFP4 (packed along HEAD_DIM)
    q_packed, q_scale = quantize_to_mxfp4(query_states.float())
    k_packed, k_scale = quantize_to_mxfp4(key_states.float())

    # Quantize V: need [B, H, D, N//2] layout (col-major, packed along N)
    v_transposed = value_states.float().permute(0, 1, 3, 2).contiguous()  # [B, H, D, N]
    v_flat = v_transposed.reshape(-1, N_padded)
    v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
    v_packed = v_packed_flat.reshape(B, H, D, N_padded // 2)
    v_scale = v_scale_flat.reshape(B, H, D, N_padded // 32)

    # Determine causal
    causal = is_causal and (M_padded == N_padded)

    # Run MXFP4 attention
    sm_scale = scale if scale is not None else (1.0 / (D ** 0.5))
    out = mxfp4_flash_attention(q_packed, k_packed, v_packed, q_scale, k_scale, v_scale,
                                 causal=causal, sm_scale=sm_scale)

    # Remove padding and convert back
    out = out[:, :, :M, :].to(query_states.dtype)
    return out


def monkey_patch_attention(model, use_mxfp4=True):
    """Monkey-patch the model's attention to use MXFP4 or restore original."""
    import transformers.models.qwen3.modeling_qwen3 as qwen3_module

    if use_mxfp4:
        # Patch the eager_attention_forward or the sdpa call
        # The simplest approach: patch F.scaled_dot_product_attention globally
        torch.nn.functional._original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = mxfp4_sdpa_replacement
        print("[PATCHED] Using MXFP4 flash attention")
    else:
        if hasattr(torch.nn.functional, '_original_sdpa'):
            torch.nn.functional.scaled_dot_product_attention = torch.nn.functional._original_sdpa
            print("[RESTORED] Using original SDPA")


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=64):
    """Generate text and return the output string + logits."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    # Get logits for the prompt (for perplexity comparison)
    outputs = model(input_ids)
    prompt_logits = outputs.logits

    # Generate
    gen_output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy for reproducibility
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


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded: {model.config.num_attention_heads} heads, head_dim={model.config.hidden_size // model.config.num_attention_heads}")
    print()

    prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"",
        "In machine learning, backpropagation is",
        "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    ]

    eval_texts = [
        "The transformer architecture consists of an encoder and a decoder. The encoder processes the input sequence and produces a set of representations. The decoder then generates the output sequence one token at a time, attending to both the encoder output and previously generated tokens.",
        "Python is a high-level programming language known for its simple syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    ]

    # --- Original SDPA ---
    print("=" * 60)
    print("ORIGINAL (FP16 SDPA)")
    print("=" * 60)
    original_outputs = []
    for prompt in prompts:
        text, logits = generate_text(model, tokenizer, prompt)
        original_outputs.append((text, logits))
        out_text = repr(text[len(prompt):][:80])
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {out_text}")
        print()

    original_ppls = []
    for text in eval_texts:
        ppl = compute_perplexity(model, tokenizer, text)
        original_ppls.append(ppl)
        print(f"  PPL: {ppl:.2f} | {text[:60]}...")

    # --- MXFP4 Attention ---
    print()
    print("=" * 60)
    print("MXFP4 (E2M1) ATTENTION")
    print("=" * 60)
    monkey_patch_attention(model, use_mxfp4=True)

    mxfp4_outputs = []
    for prompt in prompts:
        text, logits = generate_text(model, tokenizer, prompt)
        mxfp4_outputs.append((text, logits))
        out_text = repr(text[len(prompt):][:80])
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {out_text}")
        print()

    mxfp4_ppls = []
    for text in eval_texts:
        ppl = compute_perplexity(model, tokenizer, text)
        mxfp4_ppls.append(ppl)
        print(f"  PPL: {ppl:.2f} | {text[:60]}...")

    # --- Comparison ---
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Token match rate
    total_match = 0
    total_tokens = 0
    for i, prompt in enumerate(prompts):
        orig_text = original_outputs[i][0]
        mxfp4_text = mxfp4_outputs[i][0]
        orig_tokens = tokenizer.encode(orig_text)
        mxfp4_tokens = tokenizer.encode(mxfp4_text)
        min_len = min(len(orig_tokens), len(mxfp4_tokens))
        matches = sum(1 for a, b in zip(orig_tokens[:min_len], mxfp4_tokens[:min_len]) if a == b)
        total_match += matches
        total_tokens += min_len
        match_pct = matches / min_len * 100 if min_len > 0 else 0
        print(f"  Prompt {i}: token match = {matches}/{min_len} ({match_pct:.1f}%)")
        if orig_text != mxfp4_text:
            orig_snip = repr(orig_text[len(prompt):][:60])
            mxfp4_snip = repr(mxfp4_text[len(prompt):][:60])
            print(f"    ORIG:  {orig_snip}")
            print(f"    MXFP4: {mxfp4_snip}")

    print(f"\n  Overall token match: {total_match}/{total_tokens} ({total_match/total_tokens*100:.1f}%)")

    # Perplexity comparison
    print("\n  Perplexity comparison:")
    for i, text in enumerate(eval_texts):
        delta = mxfp4_ppls[i] - original_ppls[i]
        pct = delta / original_ppls[i] * 100
        print(f"    Text {i}: SDPA={original_ppls[i]:.2f}, MXFP4={mxfp4_ppls[i]:.2f}, delta={delta:+.2f} ({pct:+.1f}%)")

    # Logit cosine similarity (first prompt)
    orig_logits = original_outputs[0][1][0, -1, :]  # last token logits
    mxfp4_logits = mxfp4_outputs[0][1][0, -1, :]
    cos_sim = F.cosine_similarity(orig_logits.float().unsqueeze(0), mxfp4_logits.float().unsqueeze(0)).item()
    print(f"\n  Logit cosine similarity (prompt 0, last token): {cos_sim:.6f}")

    # Restore
    monkey_patch_attention(model, use_mxfp4=False)


if __name__ == "__main__":
    main()
