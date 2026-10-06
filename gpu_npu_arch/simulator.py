"""
Cycle-level & Analytical Hardware Simulator.
Models instruction latencies, memory hierarchy (L1/SRAM, L2, DRAM/HBM),
systolic matrix pipelines, warp vs. tile scheduling, and energy consumption.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math
from .spec import HardwareSpec, RTX_6000_ADA, TRITON_X100
from .triton_ir import TritonKernelGraph, DataType


@dataclass
class SimulationResult:
    target_name: str
    kernel_name: str
    execution_time_ms: float
    effective_tflops: float
    compute_utilization_pct: float
    dram_throughput_gb_s: float
    dram_bandwidth_utilization_pct: float
    l2_hit_rate_pct: float
    l2_throughput_tb_s: float
    energy_joules: float
    perf_per_watt: float  # TFLOPS / W
    primary_bottleneck: str
    speedup_vs_baseline: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Target": self.target_name,
            "Kernel": self.kernel_name,
            "Latency (ms)": f"{self.execution_time_ms:.4f} ms",
            "Effective TFLOPS": f"{self.effective_tflops:.2f}",
            "Compute Util (%)": f"{self.compute_utilization_pct:.1f}%",
            "DRAM BW (GB/s)": f"{self.dram_throughput_gb_s:.1f} GB/s",
            "BW Util (%)": f"{self.dram_bandwidth_utilization_pct:.1f}%",
            "L2 Hit Rate": f"{self.l2_hit_rate_pct:.1f}%",
            "Energy (J)": f"{self.energy_joules:.3f} J",
            "Perf/Watt": f"{self.perf_per_watt:.2f} TFLOPS/W",
            "Bottleneck": self.primary_bottleneck,
            "Speedup": f"{self.speedup_vs_baseline:.2f}x",
        }


class AcceleratorSimulator:
    """
    Architectural simulator capable of simulating Triton workloads
    on both NVIDIA RTX 6000 Ada and Triton-X100.
    """

    def __init__(self, spec: HardwareSpec):
        self.spec = spec

    def simulate(
        self,
        kernel: TritonKernelGraph,
        dtype: DataType = DataType.FP16,
        l2_cache_reuse_factor: float = 0.0,  # Fraction of DRAM bytes served from L2
        unaligned_memory_access: bool = False,
    ) -> SimulationResult:
        """
        Simulates the execution of a Triton kernel graph on the target hardware.
        """
        total_flops = kernel.total_flops
        nominal_bytes = kernel.total_dram_bytes

        # Determine peak TFLOPS for the given precision
        if dtype in (DataType.FP16, DataType.BF16):
            peak_tflops = self.spec.tflops_bf16
        elif dtype in (DataType.FP8_E4M3, DataType.FP8_E5M2):
            peak_tflops = self.spec.tflops_fp8
        elif dtype == DataType.FP4_MX:
            peak_tflops = self.spec.tflops_fp4 if self.spec.tflops_fp4 > 0 else self.spec.tflops_fp8
        else:
            peak_tflops = self.spec.tflops_fp32

        # -------------------------------------------------------------
        # 1. Memory Traffic & Cache Subsystem Modeling
        # -------------------------------------------------------------
        # For Triton-X100, its 256MB L2 cache (vs 96MB on Ada) dramatically
        # improves cache hit rate on larger matrix / attention working sets.
        cache_capacity_scale = self.spec.l2_cache_mb / 96.0
        effective_l2_reuse = min(0.92, l2_cache_reuse_factor * (cache_capacity_scale ** 0.5))

        # Memory traffic splitting
        bytes_from_l2 = nominal_bytes * effective_l2_reuse
        bytes_from_dram = nominal_bytes * (1.0 - effective_l2_reuse)

        # DRAM Bus Efficiency Modeling
        # RTX 6000 Ada GDDR6 suffers when access is non-unit stride or unaligned (78-85% bus efficiency)
        # Triton-X100 HBM3e with ATDE maintains 92-95% efficiency through 2D block streaming
        if self.spec.name == RTX_6000_ADA.name:
            dram_bus_efficiency = 0.78 if unaligned_memory_access else 0.86
            l2_bus_efficiency = 0.88
            # Thermal throttling under heavy sustained tensor load
            effective_clock_factor = 0.94  # Ada boosts to 2.35 GHz sustained in 300W envelope
        else:
            # Triton-X100: Native hardware tile burst via ATDE
            dram_bus_efficiency = 0.94 if unaligned_memory_access else 0.96
            l2_bus_efficiency = 0.96
            effective_clock_factor = 1.00  # High energy-efficiency tile core sustains 2.05 GHz in 350W

        effective_peak_tflops = peak_tflops * effective_clock_factor
        effective_dram_bw = self.spec.dram_bandwidth_gb_s * dram_bus_efficiency
        effective_l2_bw = (self.spec.l2_bandwidth_tb_s * 1000.0) * l2_bus_efficiency

        # Raw ideal transfer latencies
        time_dram_sec = (bytes_from_dram / (effective_dram_bw * 1e9)) if effective_dram_bw > 0 else 0.0
        time_l2_sec = (bytes_from_l2 / (effective_l2_bw * 1e9)) if effective_l2_bw > 0 else 0.0
        time_memory_sec = time_dram_sec + time_l2_sec

        # -------------------------------------------------------------
        # 2. Compute Pipeline & Tensor Unit Modeling
        # -------------------------------------------------------------
        ideal_compute_sec = (total_flops / (effective_peak_tflops * 1e12)) if effective_peak_tflops > 0 else 0.0

        # Microarchitecture overhead:
        # RTX 6000 Ada (SIMT):
        # - Register file pressure: Triton keeps large accumulator tiles in registers.
        #   On Ada SMs, this limits active warps (lowering latency hiding) -> register penalty.
        # - Shared memory bank conflict penalty during tile transposes.
        # - Warp divergence and sync barrier overhead.
        #
        # Triton-X100:
        # - Native 2D Tensor Tile Engines (no warp registers).
        # - 512KB per-TEU scratchpad with wide crossbar (no bank conflicts).
        # - Asynchronous double-buffered pipeline completely overlaps memory fetch with compute.
        if self.spec.name == RTX_6000_ADA.name:
            simt_hazard_penalty = self.spec.simt_register_file_pressure
            control_overhead = 1.0 + (self.spec.warp_scheduling_overhead_pct / 100.0)
            compute_efficiency = 0.82 / (simt_hazard_penalty * control_overhead)
            # Memory / Compute Overlap factor: Ada cp.async achieves ~65-75% overlap on Triton code
            overlap_efficiency = 0.70
        else:
            # Triton-X100: Native tile hardware with hardware DMA sequencer
            compute_efficiency = 0.95
            # Zero-overhead hardware ping-pong double buffering in 512KB TLM achieves 96% overlap
            overlap_efficiency = 0.95

        effective_compute_sec = ideal_compute_sec / compute_efficiency

        # -------------------------------------------------------------
        # 3. Overall Latency & Pipeline Overlap
        # -------------------------------------------------------------
        # Execution time is dominated by the max of compute vs memory, with partial overlap of the non-critical path
        max_path = max(effective_compute_sec, time_memory_sec)
        min_path = min(effective_compute_sec, time_memory_sec)
        unoverlapped_min_path = min_path * (1.0 - overlap_efficiency)

        total_exec_time_sec = max_path + unoverlapped_min_path
        total_exec_time_ms = total_exec_time_sec * 1000.0

        # -------------------------------------------------------------
        # 4. Metrics Derivation
        # -------------------------------------------------------------
        effective_tflops = (total_flops / total_exec_time_sec) / 1e12 if total_exec_time_sec > 0 else 0.0
        compute_utilization_pct = (effective_tflops / self.spec.tflops_bf16) * 100.0 if dtype in (DataType.FP16, DataType.BF16) else (effective_tflops / peak_tflops) * 100.0

        actual_dram_gb_s = (bytes_from_dram / total_exec_time_sec) / 1e9 if total_exec_time_sec > 0 else 0.0
        bw_utilization_pct = (actual_dram_gb_s / self.spec.dram_bandwidth_gb_s) * 100.0

        actual_l2_tb_s = (bytes_from_l2 / total_exec_time_sec) / 1e12 if total_exec_time_sec > 0 else 0.0

        # Energy consumption: Active power + baseline leakage
        # Ada: 300W TDP, active dynamic power ~280W under heavy compute, 180W under memory-bound
        # Triton-X100: 350W TDP, highly efficient tile engines
        if effective_compute_sec > time_memory_sec:
            avg_power_watts = self.spec.tdp_watts * 0.92
            bottleneck = "Compute-Bound (Tensor Cores Saturated)"
        elif abs(effective_compute_sec - time_memory_sec) / max_path < 0.15:
            avg_power_watts = self.spec.tdp_watts * 0.85
            bottleneck = "Balanced (Compute & Memory Balanced)"
        else:
            avg_power_watts = self.spec.tdp_watts * 0.68
            bottleneck = "Memory-Bound (Starved by DRAM Bandwidth)"

        energy_joules = avg_power_watts * total_exec_time_sec
        perf_per_watt = effective_tflops / avg_power_watts if avg_power_watts > 0 else 0.0

        return SimulationResult(
            target_name=self.spec.name,
            kernel_name=kernel.kernel_name,
            execution_time_ms=total_exec_time_ms,
            effective_tflops=effective_tflops,
            compute_utilization_pct=min(100.0, compute_utilization_pct),
            dram_throughput_gb_s=actual_dram_gb_s,
            dram_bandwidth_utilization_pct=min(100.0, bw_utilization_pct),
            l2_hit_rate_pct=effective_l2_reuse * 100.0,
            l2_throughput_tb_s=actual_l2_tb_s,
            energy_joules=energy_joules,
            perf_per_watt=perf_per_watt,
            primary_bottleneck=bottleneck,
        )
