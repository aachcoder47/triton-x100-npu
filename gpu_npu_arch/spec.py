"""
Hardware specifications and architectural constants for comparison:
NVIDIA RTX 6000 Ada (AD102-GL) vs. Triton-X100 (Titan-Tile NPU).
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    codename: str
    arch_type: str  # "SIMT GPU" vs "Domain-Specific Tile NPU"
    process_node: str
    die_area_mm2: float
    transistor_count_b: float
    tdp_watts: float

    # Compute Subsystem
    compute_units: int  # 142 SMs vs 64 TEUs
    core_clock_ghz: float
    tensor_cores_or_engines: int
    
    # Peak Compute (TFLOPS) - Dense / Non-sparse
    tflops_fp32: float
    tflops_fp16: float
    tflops_bf16: float
    tflops_fp8: float
    tflops_fp4: float  # Microscaling / MXFP4

    # Memory Hierarchy
    dram_capacity_gb: float
    dram_type: str  # "GDDR6 (ECC)" vs "HBM3e (ECC)"
    dram_bus_width_bits: int
    dram_bandwidth_gb_s: float
    
    # Cache Hierarchy
    l2_cache_mb: float
    l2_bandwidth_tb_s: float
    l1_sram_per_cu_kb: float  # Per SM or per TEU
    total_l1_sram_mb: float
    l1_bandwidth_tb_s: float
    
    # Microarchitectural Features
    execution_paradigm: str
    native_tile_dims: tuple  # Native hardware tensor tile dimensions (M, N, K)
    hardware_async_dma: bool  # NVIDIA TMA vs Triton Direct Asynchronous DMA (ATDE)
    simt_register_file_pressure: float  # SIMT register overhead factor (1.0 = none, >1.0 = stall penalty)
    warp_scheduling_overhead_pct: float  # % control logic overhead in SIMT
    
    @property
    def machine_balance_fp16(self) -> float:
        """Arithmetic intensity inflection point (FLOPs/byte) for FP16."""
        return (self.tflops_fp16 * 1000.0) / self.dram_bandwidth_gb_s

    @property
    def machine_balance_fp8(self) -> float:
        """Arithmetic intensity inflection point (FLOPs/byte) for FP8."""
        return (self.tflops_fp8 * 1000.0) / self.dram_bandwidth_gb_s

    @property
    def compute_energy_efficiency_fp16(self) -> float:
        """TFLOPS per Watt for FP16."""
        return self.tflops_fp16 / self.tdp_watts

    @property
    def compute_energy_efficiency_fp8(self) -> float:
        """TFLOPS per Watt for FP8."""
        return self.tflops_fp8 / self.tdp_watts


# -------------------------------------------------------------------------
# Reference 1: NVIDIA RTX 6000 Ada Generation (AD102-GL)
# -------------------------------------------------------------------------
# Official specs: 142 SMs, 18,176 CUDA Cores, 568 4th-Gen Tensor Cores.
# Clock: ~2.50 GHz boost. FP16/BF16 dense: ~728.5 TFLOPS (dense, w/o sparsity).
# FP8 dense: ~1,457 TFLOPS. Memory: 48 GB GDDR6 @ 20 Gbps over 384-bit bus = 960 GB/s.
# L2 Cache: 96 MB. L1/Shared Mem per SM: 128 KB. TDP: 300W.
RTX_6000_ADA = HardwareSpec(
    name="NVIDIA RTX 6000 Ada",
    codename="AD102-GL (Ada Lovelace)",
    arch_type="SIMT GPU (Graphics + Compute)",
    process_node="TSMC 4N (5nm class)",
    die_area_mm2=608.5,
    transistor_count_b=76.3,
    tdp_watts=300.0,
    compute_units=142,  # 142 SMs
    core_clock_ghz=2.50,
    tensor_cores_or_engines=568,  # 4x 4th-Gen Tensor Cores per SM
    tflops_fp32=91.1,
    tflops_fp16=728.5,
    tflops_bf16=728.5,
    tflops_fp8=1457.0,
    tflops_fp4=0.0,  # Not supported in Ada Lovelace (introduced in Blackwell)
    dram_capacity_gb=48.0,
    dram_type="GDDR6 ECC",
    dram_bus_width_bits=384,
    dram_bandwidth_gb_s=960.0,
    l2_cache_mb=96.0,
    l2_bandwidth_tb_s=5.12,  # AD102 L2 crossbar bandwidth
    l1_sram_per_cu_kb=128.0,  # Combined L1/Shared Memory per SM
    total_l1_sram_mb=18.176,  # 142 * 128KB
    l1_bandwidth_tb_s=18.5,
    execution_paradigm="SIMT (32-thread Warps, MMA.sync / WGMMA unrolling)",
    native_tile_dims=(16, 16, 16),  # Warp MMA instruction tile
    hardware_async_dma=False,  # Ada has async copy, but lacks Hopper/Blackwell TMA hardware tensor addresser
    simt_register_file_pressure=1.25,  # 25% stall penalty under heavy Triton register allocation
    warp_scheduling_overhead_pct=18.0,  # Warp issue & scoreboard logic area/latency
)


# -------------------------------------------------------------------------
# Target 2: Triton-X100 (Titan-Tile NPU)
# -------------------------------------------------------------------------
# Co-designed specifically for OpenAI Triton block semantics.
# Eliminates SIMT warp scheduler overhead in favor of 64 Tile Execution Units (TEUs).
# Each TEU has a native 64x64 BF16 / 128x128 FP8 Tensor Tile Matrix Engine (TTME).
# Memory: 64 GB HBM3e (3.2 TB/s bandwidth - 3.33x higher than RTX 6000 Ada).
# L2 Cache: 256 MB Ultra-Wide Tile Cache (12.8 TB/s mesh bandwidth).
# Per-TEU SRAM: 512 KB dedicated scratchpad (32 MB aggregate L1).
# Hardware Asynchronous Tile DMA Engine (ATDE) natively decodes Triton block pointers.
TRITON_X100 = HardwareSpec(
    name="Triton-X100 (Titan-Tile NPU)",
    codename="TX-100 'Triton-Silicon'",
    arch_type="Domain-Specific Tile NPU (Triton Co-Designed)",
    process_node="TSMC N3P (3nm)",
    die_area_mm2=540.0,
    transistor_count_b=92.0,
    tdp_watts=350.0,
    compute_units=64,  # 64 High-throughput Tile Execution Units (TEUs)
    core_clock_ghz=2.05,
    tensor_cores_or_engines=64,  # Native 2D/3D Tensor Engines
    tflops_fp32=131.2,
    tflops_fp16=2100.0,  # 2.88x RTX 6000 Ada
    tflops_bf16=2100.0,  # 2.88x RTX 6000 Ada
    tflops_fp8=4200.0,   # 2.88x RTX 6000 Ada
    tflops_fp4=8400.0,   # Native FP4/MXFP4 tensor support
    dram_capacity_gb=64.0,  # 4-hi or 8-hi HBM3e stack
    dram_type="HBM3e ECC (2048-bit)",
    dram_bus_width_bits=2048,
    dram_bandwidth_gb_s=3200.0,  # 3.33x RTX 6000 Ada (3.2 TB/s)
    l2_cache_mb=256.0,  # 2.67x RTX 6000 Ada (256 MB Tile Cache)
    l2_bandwidth_tb_s=12.8,  # High-radix 2D Mesh NoC
    l1_sram_per_cu_kb=512.0,  # 512 KB per TEU Tile Local Scratchpad
    total_l1_sram_mb=32.0,   # 64 * 512KB
    l1_bandwidth_tb_s=42.0,
    execution_paradigm="Tile-Dataflow (Native Triton Block-Level Dispatch)",
    native_tile_dims=(64, 64, 32),  # Native 2D matrix engine tile
    hardware_async_dma=True,  # Native Triton Block Pointer Async DMA (Zero-CPU copy)
    simt_register_file_pressure=1.0,  # No register spilling to thread-local stack
    warp_scheduling_overhead_pct=2.0,  # Block-level barrier / scoreboard
)


def get_specs_comparison_table() -> Dict[str, Any]:
    """Generates structured side-by-side metric comparison."""
    metrics = [
        ("Architecture Family", RTX_6000_ADA.arch_type, TRITON_X100.arch_type, "Domain-Specific Tile"),
        ("Process Technology", RTX_6000_ADA.process_node, TRITON_X100.process_node, "Next-gen 3nm"),
        ("TDP (Watts)", f"{RTX_6000_ADA.tdp_watts} W", f"{TRITON_X100.tdp_watts} W", "+16.7% Power Budget"),
        ("Compute Units", f"{RTX_6000_ADA.compute_units} SMs", f"{TRITON_X100.compute_units} TEUs", "Wide-Tile Units"),
        ("Clock Frequency", f"{RTX_6000_ADA.core_clock_ghz:.2f} GHz", f"{TRITON_X100.core_clock_ghz:.2f} GHz", "Optimized Iso-power"),
        ("Dense FP16/BF16 Compute", f"{RTX_6000_ADA.tflops_bf16:.1f} TFLOPS", f"{TRITON_X100.tflops_bf16:.1f} TFLOPS", f"{TRITON_X100.tflops_bf16 / RTX_6000_ADA.tflops_bf16:.2f}x"),
        ("Dense FP8 Compute", f"{RTX_6000_ADA.tflops_fp8:.1f} TFLOPS", f"{TRITON_X100.tflops_fp8:.1f} TFLOPS", f"{TRITON_X100.tflops_fp8 / RTX_6000_ADA.tflops_fp8:.2f}x"),
        ("Dense FP4 / MXFP4 Compute", "N/A (0.0 TFLOPS)", f"{TRITON_X100.tflops_fp4:.1f} TFLOPS", "Native MXFP4 Engine"),
        ("DRAM Capacity", f"{RTX_6000_ADA.dram_capacity_gb:.0f} GB GDDR6", f"{TRITON_X100.dram_capacity_gb:.0f} GB HBM3e", "+33.3% Capacity"),
        ("Memory Bus Width", f"{RTX_6000_ADA.dram_bus_width_bits}-bit", f"{TRITON_X100.dram_bus_width_bits}-bit", "5.33x Bus Width"),
        ("DRAM Bandwidth", f"{RTX_6000_ADA.dram_bandwidth_gb_s:.1f} GB/s", f"{TRITON_X100.dram_bandwidth_gb_s:.1f} GB/s", f"{TRITON_X100.dram_bandwidth_gb_s / RTX_6000_ADA.dram_bandwidth_gb_s:.2f}x Bandwidth"),
        ("L2 Cache Capacity", f"{RTX_6000_ADA.l2_cache_mb:.0f} MB", f"{TRITON_X100.l2_cache_mb:.0f} MB", f"{TRITON_X100.l2_cache_mb / RTX_6000_ADA.l2_cache_mb:.2f}x L2 Cache"),
        ("L2 Cache Bandwidth", f"{RTX_6000_ADA.l2_bandwidth_tb_s:.2f} TB/s", f"{TRITON_X100.l2_bandwidth_tb_s:.2f} TB/s", f"{TRITON_X100.l2_bandwidth_tb_s / RTX_6000_ADA.l2_bandwidth_tb_s:.2f}x L2 Bandwidth"),
        ("L1 / Scratchpad per CU", f"{RTX_6000_ADA.l1_sram_per_cu_kb:.0f} KB (Shared)", f"{TRITON_X100.l1_sram_per_cu_kb:.0f} KB (Tile Scratchpad)", "4.0x SRAM per Unit"),
        ("Total L1 SRAM on Die", f"{RTX_6000_ADA.total_l1_sram_mb:.1f} MB", f"{TRITON_X100.total_l1_sram_mb:.1f} MB", "1.76x Total L1"),
        ("Execution Primitive", "32-Thread SIMT Warps", "Triton 2D/3D Tensor Blocks", "Hardware Native Tiles"),
        ("Async DMA Engine", "Basic cp.async (SIMT)", "Hardware ATDE (Triton Block Ptr)", "Zero-CPU Tensor DMA"),
        ("FP16 Machine Balance", f"{RTX_6000_ADA.machine_balance_fp16:.1f} FLOPs/B", f"{TRITON_X100.machine_balance_fp16:.1f} FLOPs/B", "Lower = Better Balanced"),
        ("FP8 Machine Balance", f"{RTX_6000_ADA.machine_balance_fp8:.1f} FLOPs/B", f"{TRITON_X100.machine_balance_fp8:.1f} FLOPs/B", "Lower = Less Memory Stalled"),
        ("FP16 Energy Efficiency", f"{RTX_6000_ADA.compute_energy_efficiency_fp16:.2f} TFLOPS/W", f"{TRITON_X100.compute_energy_efficiency_fp16:.2f} TFLOPS/W", f"{TRITON_X100.compute_energy_efficiency_fp16 / RTX_6000_ADA.compute_energy_efficiency_fp16:.2f}x Perf/Watt"),
        ("FP8 Energy Efficiency", f"{RTX_6000_ADA.compute_energy_efficiency_fp8:.2f} TFLOPS/W", f"{TRITON_X100.compute_energy_efficiency_fp8:.2f} TFLOPS/W", f"{TRITON_X100.compute_energy_efficiency_fp8 / RTX_6000_ADA.compute_energy_efficiency_fp8:.2f}x Perf/Watt"),
    ]
    return {"headers": ["Metric", "NVIDIA RTX 6000 Ada", "Triton-X100 (Titan-Tile NPU)", "Advantage"], "rows": metrics}
