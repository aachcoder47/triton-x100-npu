"""
Unit tests for the hardware specifications, Triton IR, and simulation models.
"""

import unittest
from gpu_npu_arch.spec import RTX_6000_ADA, TRITON_X100
from gpu_npu_arch.triton_ir import DataType
from gpu_npu_arch.benchmarks import build_triton_gemm_kernel, build_flash_attention_kernel
from gpu_npu_arch.simulator import AcceleratorSimulator


class TestHardwareCoDesign(unittest.TestCase):

    def test_hardware_specs_integrity(self):
        self.assertGreater(TRITON_X100.dram_bandwidth_gb_s, RTX_6000_ADA.dram_bandwidth_gb_s)
        self.assertGreater(TRITON_X100.tflops_bf16, RTX_6000_ADA.tflops_bf16)
        self.assertGreater(TRITON_X100.l2_cache_mb, RTX_6000_ADA.l2_cache_mb)
        self.assertGreater(TRITON_X100.compute_energy_efficiency_fp16, RTX_6000_ADA.compute_energy_efficiency_fp16)

    def test_gemm_kernel_graph(self):
        kernel = build_triton_gemm_kernel(4096, 4096, 4096, block_m=128, block_n=128, block_k=64)
        self.assertGreater(kernel.total_flops, 0)
        self.assertGreater(kernel.arithmetic_intensity, 10.0)

    def test_simulation_speedup(self):
        kernel = build_triton_gemm_kernel(4096, 4096, 4096)
        ada_sim = AcceleratorSimulator(RTX_6000_ADA)
        tx_sim = AcceleratorSimulator(TRITON_X100)

        ada_res = ada_sim.simulate(kernel, dtype=DataType.BF16)
        tx_res = tx_sim.simulate(kernel, dtype=DataType.BF16)

        # Triton-X100 must be faster than RTX 6000 Ada
        self.assertLess(tx_res.execution_time_ms, ada_res.execution_time_ms)
        speedup = ada_res.execution_time_ms / tx_res.execution_time_ms
        self.assertGreater(speedup, 1.5)

    def test_flash_attention_simulation(self):
        kernel = build_flash_attention_kernel(batch_size=2, num_heads=32, seq_len=4096, head_dim=128)
        ada_sim = AcceleratorSimulator(RTX_6000_ADA)
        tx_sim = AcceleratorSimulator(TRITON_X100)

        ada_res = ada_sim.simulate(kernel, dtype=DataType.BF16, l2_cache_reuse_factor=0.5)
        tx_res = tx_sim.simulate(kernel, dtype=DataType.BF16, l2_cache_reuse_factor=0.5)

        self.assertLess(tx_res.execution_time_ms, ada_res.execution_time_ms)


if __name__ == "__main__":
    unittest.main()
