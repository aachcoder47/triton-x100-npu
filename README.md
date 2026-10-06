# Triton-X100 (Titan-Tile NPU): Hardware Co-Designed for OpenAI Triton

A domain-specific AI accelerator architecture and cycle/roofline simulation suite designed from first principles to outperform the **NVIDIA RTX 6000 Ada Generation (AD102-GL)** across modern Deep Learning and LLM workloads.

---

## 🚀 Key Architectural Innovations

| Feature | NVIDIA RTX 6000 Ada | Triton-X100 (Titan-Tile NPU) | Architectural Why |
| :--- | :--- | :--- | :--- |
| **Execution Paradigm** | 32-Thread SIMT Warps | **Native 2D/3D Tile-Dataflow** | Eliminates warp divergence, `ldmatrix` shuffles, and SIMT register file pressure. |
| **DRAM Memory Subsystem** | 48 GB GDDR6 (960 GB/s) | **64 GB HBM3e (3,200 GB/s)** | **3.33x higher bandwidth** resolves the memory wall in attention decoding and RMSNorm. |
| **Dense Compute (BF16 / FP8)** | 728.5 TF / 1,457 TF | **2,100 TF / 4,200 TF** | **2.88x higher dense tensor throughput** via 64 TEUs running native systolic tile engines. |
| **On-Chip L2 Tile Cache** | 96 MB (5.12 TB/s) | **256 MB (12.80 TB/s)** | **2.67x larger L2** retains FlashAttention KV working sets on-chip without hitting DRAM. |
| **L1 SRAM per Compute Unit** | 128 KB (Shared) | **512 KB Tile Local Scratchpad** | **4.0x larger scratchpad** allows deep 8-stage software/hardware pipelining. |
| **Async DMA Mechanism** | `cp.async` (Scalar addresses) | **Hardware ATDE** | Direct hardware mapping of `tl.make_block_ptr` & `tl.advance` without scalar CPU overhead. |
| **FP4 Support** | None (Ada has no FP4) | **Native MXFP4 (8,400 TFLOPS)** | 5.76x speedup on next-generation microscaling quantized LLMs. |

---

## 📊 Benchmark Summary: Speedups over RTX 6000 Ada

| Benchmark Kernel | RTX 6000 Ada Latency | Triton-X100 Latency | Speedup | Energy Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **GEMM BF16 (4096³)** | 2.426 ms | **0.583 ms** | **4.16x** | 72.0% |
| **GEMM FP8 (8192³)** | 6.708 ms | **1.484 ms** | **4.52x** | 74.2% |
| **FlashAttention-2 BF16 (Seq 4K)** | 6.257 ms | **0.941 ms** | **6.65x** | 82.5% |
| **FlashAttention-3 FP8 (Seq 8K)** | 5.356 ms | **0.887 ms** | **6.03x** | 80.7% |
| **Fused RMSNorm + SwiGLU** | 0.601 ms | **0.143 ms** | **4.20x** | 72.2% |
| **LLM Decode Attention (KV Stream)** | 0.624 ms | **0.164 ms** | **3.80x** | 69.3% |

---

## 🛠 Usage & CLI Commands

### 1. View Architectural Specifications Side-by-Side
```powershell
python -m gpu_npu_arch.cli --specs
```

### 2. Run Comparative Triton Benchmark Suite
```powershell
python -m gpu_npu_arch.cli --benchmark
```

### 3. Inspect Triton IR to Hardware Lowering
```powershell
python -m gpu_npu_arch.cli --lowering
```

### 4. Generate Visual Plots & Full Technical Report
```powershell
python -m gpu_npu_arch.cli --report --outdir artifacts
```

### 5. Run the Silicon EDA Flow (Simulation & ASIC Gate Synthesis)
```powershell
python eda/run_eda_flow.py
```
This runs:
1. **Functional Verification (Icarus Verilog)**: Simulates the 2D Systolic Array and Asynchronous Tile DMA engine.
2. **ASIC Logic Synthesis (Yosys)**: Translates the RTL into standard cells, producing `eda/synthesized_netlist.v` (13,134 standard cells, 807 flip-flops).
3. **Signoff Telemetry**: Generates `artifacts/EDA_SYNTHESIS_SIGNOFF.md`.

### 6. Run Unit Tests
```powershell
python -m unittest discover tests
```

---

## 📁 Repository Structure
```
gpu/
├── gpu_npu_arch/
│   ├── __init__.py
│   ├── spec.py               # Hardware specs (RTX 6000 Ada vs Triton-X100)
│   ├── triton_ir.py          # OpenAI Triton IR block dialect & lowering models
│   ├── simulator.py          # Cycle-level & roofline hardware simulator
│   ├── benchmarks.py         # Production AI workloads (GEMM, FlashAttn, RMSNorm, Decode)
│   ├── visualizer.py         # Matplotlib roofline & speedup bar chart generators
│   ├── report_generator.py   # Benchmark automation & report compiler
│   └── cli.py                # Rich terminal user interface
├── tests/
│   └── test_simulation.py    # Unit tests for specs, IR, and simulation
├── artifacts/
│   ├── roofline_model.png
│   ├── speedup_comparison.png
│   └── TRITON_X100_VS_RTX6000_ADA_REPORT.md
└── README.md
```
