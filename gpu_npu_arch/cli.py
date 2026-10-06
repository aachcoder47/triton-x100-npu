"""
Command Line Interface for Triton-X100 Architecture Simulator.
Allows running comparative benchmarks, viewing specs, and generating visual reports.
"""

import argparse
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .spec import get_specs_comparison_table
from .report_generator import run_benchmark_suite, generate_full_report
from .triton_ir import get_triton_lowering_mechanisms


console = Console()


def display_specs():
    specs_data = get_specs_comparison_table()
    table = Table(title="[bold cyan]Architectural Specifications: NVIDIA RTX 6000 Ada vs. Triton-X100 (Titan-Tile NPU)[/bold cyan]")
    for h in specs_data["headers"]:
        table.add_column(h, style="bold" if h == "Metric" else "white")

    for row in specs_data["rows"]:
        color = "green" if "x" in row[3] or "Native" in row[3] or "+" in row[3] else "yellow"
        table.add_row(row[0], row[1], row[2], f"[{color}]{row[3]}[/{color}]")

    console.print(table)


def display_benchmarks():
    console.print(Panel("[bold green]Running OpenAI Triton AI Benchmark Suite on RTX 6000 Ada and Triton-X100...[/bold green]"))
    benchmark_pairs = run_benchmark_suite()

    table = Table(title="[bold cyan]Benchmark Simulation Results: Latency, Throughput & Speedup[/bold cyan]")
    table.add_column("Triton Kernel", style="cyan")
    table.add_column("RTX 6000 Ada", style="magenta")
    table.add_column("Triton-X100", style="green")
    table.add_column("Speedup", style="bold green")
    table.add_column("Ada TFLOPS", style="dim")
    table.add_column("TX-100 TFLOPS", style="bold yellow")
    table.add_column("Bottleneck", style="italic")

    for ada_res, tx_res in benchmark_pairs:
        speedup = ada_res.execution_time_ms / tx_res.execution_time_ms
        table.add_row(
            ada_res.kernel_name,
            f"{ada_res.execution_time_ms:.3f} ms",
            f"{tx_res.execution_time_ms:.3f} ms",
            f"{speedup:.2f}x",
            f"{ada_res.effective_tflops:.1f}",
            f"{tx_res.effective_tflops:.1f}",
            tx_res.primary_bottleneck,
        )

    console.print(table)


def display_lowering():
    console.print(Panel("[bold yellow]OpenAI Triton Dialect: SIMT Lowering (Ada) vs Native Tile Lowering (Triton-X100)[/bold yellow]"))
    mechanisms = get_triton_lowering_mechanisms()

    for m in mechanisms:
        console.print(f"[bold cyan]Operation:[/bold cyan] [underline]{m.triton_op}[/underline]")
        console.print(f"  [red]RTX 6000 Ada (SIMT):[/red] {m.rtx_6000_ada_lowering}")
        console.print(f"  [red]  Penalty:[/red] {m.rtx_6000_ada_penalties}")
        console.print(f"  [green]Triton-X100 (Native Tile):[/green] {m.triton_x100_lowering}")
        console.print(f"  [green]  Advantage:[/green] {m.triton_x100_advantages}")
        console.print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="Triton-X100 Hardware Co-Design & Simulator CLI")
    parser.add_argument("--specs", action="store_true", help="Display architectural comparison specs")
    parser.add_argument("--benchmark", action="store_true", help="Run Triton kernel benchmark suite")
    parser.add_argument("--lowering", action="store_true", help="Display Triton IR compiler lowering analysis")
    parser.add_argument("--report", action="store_true", help="Generate full markdown report and charts")
    parser.add_argument("--outdir", default="artifacts", help="Output directory for reports and charts")

    args = parser.parse_args()

    # If no flags passed, run everything
    if not any([args.specs, args.benchmark, args.lowering, args.report]):
        display_specs()
        console.print("\n")
        display_lowering()
        console.print("\n")
        display_benchmarks()
        console.print("\n[bold green]Generating visual plots and technical report...[/bold green]")
        paths = generate_full_report(args.outdir)
        console.print(f"[bold cyan]Report generated:[/bold cyan] {paths['report_md']}")
        console.print(f"[bold cyan]Roofline plot:[/bold cyan] {paths['roofline_png']}")
        console.print(f"[bold cyan]Speedup plot:[/bold cyan] {paths['speedup_png']}")
        return

    if args.specs:
        display_specs()
    if args.lowering:
        display_lowering()
    if args.benchmark:
        display_benchmarks()
    if args.report:
        paths = generate_full_report(args.outdir)
        console.print(f"[bold cyan]Report generated:[/bold cyan] {paths['report_md']}")
        console.print(f"[bold cyan]Roofline plot:[/bold cyan] {paths['roofline_png']}")
        console.print(f"[bold cyan]Speedup plot:[/bold cyan] {paths['speedup_png']}")


if __name__ == "__main__":
    main()
