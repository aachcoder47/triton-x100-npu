"""
Roofline Model & Benchmark Visualizer.
Generates publication-quality charts comparing NVIDIA RTX 6000 Ada and Triton-X100.
"""

import os
from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

from .spec import RTX_6000_ADA, TRITON_X100
from .simulator import SimulationResult


def plot_roofline_comparison(output_path: str) -> str:
    """
    Plots the operational Roofline model for both RTX 6000 Ada and Triton-X100 (Titan-Tile NPU),
    highlighting the memory-bound vs compute-bound inflection points.
    """
    intensities = np.logspace(-1, 4, 500)  # 0.1 to 10,000 FLOPs/byte

    # RTX 6000 Ada Rooflines (FP16/BF16 and FP8)
    # Peak Bandwidth: 960 GB/s = 0.96 TB/s
    ada_bw = RTX_6000_ADA.dram_bandwidth_gb_s / 1000.0  # TB/s
    ada_fp16_ceil = RTX_6000_ADA.tflops_bf16
    ada_fp8_ceil = RTX_6000_ADA.tflops_fp8

    ada_fp16_perf = np.minimum(ada_fp16_ceil, intensities * ada_bw)
    ada_fp8_perf = np.minimum(ada_fp8_ceil, intensities * ada_bw)

    # Triton-X100 Rooflines (FP16/BF16 and FP8)
    # Peak Bandwidth: 3200 GB/s = 3.20 TB/s
    tx_bw = TRITON_X100.dram_bandwidth_gb_s / 1000.0  # TB/s
    tx_fp16_ceil = TRITON_X100.tflops_bf16
    tx_fp8_ceil = TRITON_X100.tflops_fp8

    tx_fp16_perf = np.minimum(tx_fp16_ceil, intensities * tx_bw)
    tx_fp8_perf = np.minimum(tx_fp8_ceil, intensities * tx_bw)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Plot RTX 6000 Ada curves
    ax.plot(intensities, ada_fp16_perf, label=f"RTX 6000 Ada FP16 (Peak: {ada_fp16_ceil:.0f} TF, 960 GB/s)", color="#38bdf8", linestyle="--", linewidth=2.2)
    ax.plot(intensities, ada_fp8_perf, label=f"RTX 6000 Ada FP8 (Peak: {ada_fp8_ceil:.0f} TF, 960 GB/s)", color="#818cf8", linestyle=":", linewidth=2.0)

    # Plot Triton-X100 curves
    ax.plot(intensities, tx_fp16_perf, label=f"Triton-X100 FP16 (Peak: {tx_fp16_ceil:.0f} TF, 3.2 TB/s)", color="#22c55e", linewidth=2.8)
    ax.plot(intensities, tx_fp8_perf, label=f"Triton-X100 FP8 (Peak: {tx_fp8_ceil:.0f} TF, 3.2 TB/s)", color="#eab308", linewidth=2.8)

    # Annotate Key Workloads
    workloads = [
        ("LLM Decode\n(KV Cache)", 1.0, 3.2, "#f43f5e"),
        ("RMSNorm+\nSwiGLU", 4.2, 13.4, "#fb923c"),
        ("FlashAttn\n(Tiled)", 64.0, 204.8, "#a855f7"),
        ("Dense GEMM\n(Large)", 512.0, 1638.4, "#06b6d4"),
    ]

    for name, x, y_approx, color in workloads:
        ax.axvline(x=x, color=color, linestyle="--", alpha=0.4, linewidth=1.2)
        ax.scatter([x], [min(tx_fp16_ceil, x * tx_bw)], color=color, s=70, zorder=5)
        ax.annotate(
            name,
            xy=(x, min(tx_fp16_ceil, x * tx_bw) * 1.3),
            fontsize=9,
            fontweight="bold",
            color=color,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor=color, alpha=0.85),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 5000)
    ax.set_ylim(0.5, 6000)

    ax.set_xlabel("Operational Intensity (FLOPs / Byte of DRAM Traffic)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax.set_ylabel("Attainable Performance (TFLOPS)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax.set_title("Architectural Roofline Comparison: NVIDIA RTX 6000 Ada vs. Triton-X100 NPU", fontsize=14, fontweight="bold", color="#f8fafc", pad=15)

    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.grid(True, which="both", color="#334155", linestyle=":", alpha=0.6)
    legend = ax.legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def plot_speedup_barchart(
    results_pairs: List[Tuple[SimulationResult, SimulationResult]],
    output_path: str,
) -> str:
    """
    Plots side-by-side performance comparison across simulated workloads.
    """
    labels = []
    speedups = []
    ada_latencies = []
    tx_latencies = []

    for ada_res, tx_res in results_pairs:
        labels.append(ada_res.kernel_name.replace("flash_attention_", "FA_").replace("triton_gemm_", "GEMM_"))
        speedups.append(ada_res.execution_time_ms / tx_res.execution_time_ms)
        ada_latencies.append(ada_res.execution_time_ms)
        tx_latencies.append(tx_res.execution_time_ms)

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(13, 6.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")

    bars = ax1.bar(x, speedups, width=0.45, color="#10b981", edgecolor="#34d399", linewidth=1.5, zorder=3)

    # Annotate speedup values on top of bars
    for bar, sp in zip(bars, speedups):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.08,
            f"{sp:.2f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#a7f3d0",
        )

    ax1.axhline(1.0, color="#f43f5e", linestyle="--", linewidth=1.5, label="RTX 6000 Ada Baseline (1.0x)")
    ax1.set_ylabel("Speedup vs NVIDIA RTX 6000 Ada", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax1.set_title("Triton-X100 Speedup across OpenAI Triton AI Workloads", fontsize=14, fontweight="bold", color="#f8fafc", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9, color="#cbd5e1", fontweight="bold")
    ax1.set_ylim(0, max(speedups) * 1.25)
    ax1.tick_params(colors="#cbd5e1")
    ax1.grid(axis="y", color="#334155", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f1f5f9")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path
