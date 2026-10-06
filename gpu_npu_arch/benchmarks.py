"""
Production AI Workloads modeled as Triton Kernel Graphs.
Includes:
- Pipelined GEMM (BF16 & FP8)
- FlashAttention-2 / FlashAttention-3 (Tiled with Online Softmax)
- Fused RMSNorm + SwiGLU (LLM Feed-Forward Block)
- LLM Generative Decode Attention (KV-Cache Streaming)
"""

from typing import List, Tuple
from .triton_ir import TritonKernelGraph, TritonOp, TileShape, DataType


def build_triton_gemm_kernel(
    m: int,
    n: int,
    k: int,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    dtype: DataType = DataType.BF16,
) -> TritonKernelGraph:
    """
    Constructs a Triton Block GEMM Kernel Graph:
    Grid: (ceil(M / BLOCK_M), ceil(N / BLOCK_N), 1)
    Inside each block:
      Loop over K with step BLOCK_K:
        a_tile = tl.load(a_ptr)
        b_tile = tl.load(b_ptr)
        acc += tl.dot(a_tile, b_tile)
      tl.store(c_ptr, acc)
    """
    grid_x = (m + block_m - 1) // block_m
    grid_y = (n + block_n - 1) // block_n
    grid_z = 1
    num_k_steps = (k + block_k - 1) // block_k

    kernel = TritonKernelGraph(
        kernel_name=f"triton_gemm_{dtype.value.upper()}_{m}x{n}x{k}",
        grid_dim=(grid_x, grid_y, grid_z),
        description=f"Triton Block GEMM: M={m}, N={n}, K={k}, Block=[{block_m}x{block_n}x{block_k}], DType={dtype.value}",
    )

    tile_a = TileShape((block_m, block_k), dtype)
    tile_b = TileShape((block_k, block_n), dtype)
    tile_c = TileShape((block_m, block_n), dtype)

    # Accumulate operations across the K-dimension loop
    for _ in range(num_k_steps):
        kernel.add_op(TritonOp.make_load(tile_a, memory_level="DRAM"))
        kernel.add_op(TritonOp.make_load(tile_b, memory_level="DRAM"))
        kernel.add_op(TritonOp.make_dot(tile_a, tile_b, acc_dtype=DataType.FP32))

    # Store final accumulator tile to global memory
    kernel.add_op(TritonOp.make_store(tile_c, memory_level="DRAM"))
    return kernel


def build_flash_attention_kernel(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int = 128,
    block_m: int = 64,
    block_n: int = 64,
    dtype: DataType = DataType.BF16,
) -> TritonKernelGraph:
    """
    Constructs a Triton FlashAttention-2 / 3 Kernel Graph with Online Softmax:
    Grid: (ceil(seq_len / BLOCK_M), batch_size * num_heads, 1)
    For each Q block:
      Load Q tile from DRAM/L2
      Loop over K, V blocks (seq_len / BLOCK_N):
        Load K tile
        S = tl.dot(Q, K^T) * scale
        Online softmax correction (tl.max, tl.exp, tl.sum)
        Load V tile
        O = tl.dot(P, V)
      Store O tile to DRAM/L2
    """
    grid_m = (seq_len + block_m - 1) // block_m
    grid_bh = batch_size * num_heads
    num_kv_steps = (seq_len + block_n - 1) // block_n

    kernel = TritonKernelGraph(
        kernel_name=f"flash_attention_{dtype.value.upper()}_B{batch_size}_H{num_heads}_S{seq_len}_D{head_dim}",
        grid_dim=(grid_m, grid_bh, 1),
        description=f"FlashAttention Tiled Kernel: Batch={batch_size}, Heads={num_heads}, SeqLen={seq_len}, Dim={head_dim}",
    )

    tile_q = TileShape((block_m, head_dim), dtype)
    tile_k = TileShape((head_dim, block_n), dtype)
    tile_v = TileShape((block_n, head_dim), dtype)
    tile_o = TileShape((block_m, head_dim), dtype)
    tile_s = TileShape((block_m, block_n), DataType.FP32)

    # Initial Q load (loaded once per Q-block)
    kernel.add_op(TritonOp.make_load(tile_q, memory_level="DRAM/L2"))

    # Iterative K, V loop with online softmax
    for _ in range(num_kv_steps):
        kernel.add_op(TritonOp.make_load(tile_k, memory_level="DRAM/L2"))
        kernel.add_op(TritonOp.make_dot(tile_q, tile_k, acc_dtype=DataType.FP32))
        # Online softmax elementwise rescale & exp
        kernel.add_op(TritonOp.make_elementwise(tile_s, op_name="online_softmax_rescale", flops_per_elem=4))
        kernel.add_op(TritonOp.make_load(tile_v, memory_level="DRAM/L2"))
        kernel.add_op(TritonOp.make_dot(tile_s, tile_v, acc_dtype=DataType.FP32))

    # Write output tile
    kernel.add_op(TritonOp.make_store(tile_o, memory_level="DRAM/L2"))
    return kernel


def build_fused_rmsnorm_swiglu_kernel(
    batch_tokens: int,  # e.g., 4096 tokens
    hidden_dim: int = 8192,
    intermediate_dim: int = 21760,  # LLaMA 3 70B intermediate dim
    block_size: int = 1024,
    dtype: DataType = DataType.BF16,
) -> TritonKernelGraph:
    """
    Constructs a Triton Fused RMSNorm + SwiGLU activation block.
    This kernel is heavily MEMORY-BANDWIDTH BOUND (Arithmetic Intensity ~ 2 to 6 FLOPs/byte).
    Computes:
      RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
      SwiGLU(x) = (x @ W_gate * silu(x @ W_gate)) * (x @ W_up)
    """
    grid_x = batch_tokens
    grid_y = 1
    grid_z = 1

    kernel = TritonKernelGraph(
        kernel_name=f"fused_rmsnorm_swiglu_{batch_tokens}_x_{hidden_dim}",
        grid_dim=(grid_x, grid_y, grid_z),
        description=f"Fused RMSNorm & SwiGLU: Tokens={batch_tokens}, Hidden={hidden_dim}, Intermediate={intermediate_dim}",
    )

    # Memory traffic per token:
    # Read input token (hidden_dim * 2 bytes)
    # Read RMSNorm weight (hidden_dim * 2 bytes)
    # Compute RMSNorm (3 FLOPs/elem)
    # Read projection activations (2 * intermediate_dim * 2 bytes)
    # Compute SiLU + Elementwise Mul (5 FLOPs/elem)
    # Write intermediate output (intermediate_dim * 2 bytes)
    tile_in = TileShape((1, hidden_dim), dtype)
    tile_weights = TileShape((1, hidden_dim), dtype)
    tile_proj = TileShape((1, intermediate_dim), dtype)

    kernel.add_op(TritonOp.make_load(tile_in))
    kernel.add_op(TritonOp.make_load(tile_weights))
    kernel.add_op(TritonOp.make_elementwise(tile_in, op_name="rmsnorm_stat", flops_per_elem=4))
    kernel.add_op(TritonOp.make_load(tile_proj))
    kernel.add_op(TritonOp.make_elementwise(tile_proj, op_name="swiglu_silu_mul", flops_per_elem=6))
    kernel.add_op(TritonOp.make_store(tile_proj))
    return kernel


def build_llm_decode_attention_kernel(
    batch_size: int = 8,
    num_heads: int = 32,
    context_length: int = 4096,
    head_dim: int = 128,
    dtype: DataType = DataType.FP16,
) -> TritonKernelGraph:
    """
    Constructs an LLM Generative Decoding Attention kernel (KV-Cache lookup).
    During generation, 1 new Query token per sequence attends to all previous K, V tokens.
    Arithmetic Intensity is extremely low (~1.0 FLOPs/byte).
    Memory bandwidth is the single overriding bottleneck!
    """
    grid_b = batch_size
    grid_h = num_heads
    grid_z = 1

    kernel = TritonKernelGraph(
        kernel_name=f"llm_decode_attention_B{batch_size}_H{num_heads}_CTX{context_length}",
        grid_dim=(grid_b, grid_h, grid_z),
        description=f"LLM Decode Attention: Batch={batch_size}, Heads={num_heads}, Context={context_length}, Dim={head_dim}",
    )

    # For each sequence & head:
    # Read 1 Query vector: 1 x 128 x 2 bytes
    # Stream entire KV cache: 2 * context_length * 128 * 2 bytes
    # Flops: 2 * (context_length * head_dim) [Q*K^T] + 2 * (context_length * head_dim) [P*V]
    tile_q = TileShape((1, head_dim), dtype)
    tile_kv_cache = TileShape((context_length, head_dim), dtype)
    tile_out = TileShape((1, head_dim), dtype)

    kernel.add_op(TritonOp.make_load(tile_q))
    kernel.add_op(TritonOp.make_load(tile_kv_cache))  # Stream K
    kernel.add_op(TritonOp.make_load(tile_kv_cache))  # Stream V
    # Flop count: 4 * context_length * head_dim
    flops = 4 * context_length * head_dim
    dot_op = TritonOp(
        op_type=TritonOp.make_dot(TileShape((1, head_dim), dtype), TileShape((head_dim, context_length), dtype)).op_type,
        name="tl.dot_decode_attention",
        output_shape=tile_out,
        input_shapes=[tile_q, tile_kv_cache],
        flop_count=flops,
        bytes_transferred=0,
        is_compute_op=True,
    )
    kernel.add_op(dot_op)
    kernel.add_op(TritonOp.make_store(tile_out))
    return kernel
