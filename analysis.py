#!/usr/bin/env python3
"""
Analysis and Comparison of CFR Solvers for Kuhn Poker.

This script implements the following analysis plan:
1. Run all solvers for a fixed number of iterations (T).
2. Calculate the Exploitability of the strategy profile at intervals of t.
3. Track memory usage (estimated) for each solver.
4. Generate comparative graphs plotting:
   - Exploitability (y-axis) vs. Iterations (x-axis)
   - Exploitability (y-axis) vs. Memory Usage (x-axis)
"""

import argparse
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
from typing import List, Tuple, Dict, Any

from kuhn_poker.solvers import (
    VanillaCFR, CFRPlus, DiscountedCFR, LinearCFR,
    QuadraticCFR, ExponentialCFR, SoftmaxCFR,
    RandomSolver, PrunedCFR
)

def get_memory_usage(obj):
    """
    Estimate memory usage of a Python object.
    This is a rough estimation.
    """
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum([get_memory_usage(v) for v in obj.values()])
        size += sum([get_memory_usage(k) for k in obj.keys()])
    elif isinstance(obj, list):
        size += sum([get_memory_usage(i) for i in obj])
    elif hasattr(obj, '__dict__'):
        size += get_memory_usage(obj.__dict__)
    elif hasattr(obj, '__slots__'): # handle slots if present
        size += sum([get_memory_usage(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s)])
    return size

def run_analysis(iterations: int, interval: int, seed: int = 42, output_dir: str = "results"):
    """
    Run the analysis for all solvers.
    
    Args:
        iterations (T): Total number of iterations to run.
        interval (t): Interval at which to calculate exploitability.
        seed: Random seed for reproducibility.
        output_dir: Directory to save results and plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*70}")
    print(f"  Comparative Analysis of CFR Solvers")
    print(f"  Iterations (T): {iterations}")
    print(f"  Interval (t): {interval}")
    print(f"{'='*70}\n")

    # Initialize all solvers
    solvers = [
        ("Vanilla CFR", VanillaCFR(seed=seed)),
        ("CFR+", CFRPlus(alternating=True, seed=seed)),
        ("Discounted CFR", DiscountedCFR(alpha=1.5, beta=0.5, gamma=2.0, seed=seed)),
        ("Linear CFR", LinearCFR(seed=seed)),
        ("Quadratic CFR", QuadraticCFR(seed=seed)),
        ("Exponential CFR", ExponentialCFR(rate=0.001, seed=seed)),
        ("Softmax CFR", SoftmaxCFR(temperature=0.1, seed=seed)),
        ("Random Solver", RandomSolver(seed=seed)),
        ("Pruned CFR", PrunedCFR(pruning_threshold=0.01, seed=seed)),
    ]

    results = []

    for name, solver in solvers:
        print(f"Running {name}...")
        
        exploitability_history = []
        memory_usage_history = [] # Track memory usage at each interval
        
        def callback(iteration, expl):
            exploitability_history.append((iteration, expl))
            
            # Estimate memory usage
            # We focus on the solver object itself which contains the info sets
            mem_usage_bytes = get_memory_usage(solver)
            mem_usage_mb = mem_usage_bytes / (1024 * 1024)
            memory_usage_history.append((iteration, mem_usage_mb))

            # Optional: print progress less frequently to avoid clutter
            if iteration % (interval * 10) == 0:
                 print(f"  Iter {iteration}: {expl:.6f} (Mem: {mem_usage_mb:.4f} MB)")

        solver.train(
            iterations,
            callback=callback,
            compute_exploitability_every=interval
        )
        
        final_expl = solver.compute_exploitability()
        final_mem = get_memory_usage(solver) / (1024 * 1024)
        
        print(f"  Final exploitability: {final_expl:.6f}")
        print(f"  Final memory usage: {final_mem:.6f} MB\n")
        
        results.append({
            "name": name,
            "final_exploitability": final_expl,
            "final_memory_mb": final_mem,
            "history": exploitability_history,
            "memory_history": memory_usage_history
        })

    # Generate Comparative Graphs
    generate_plots(results, output_dir)
    print_summary(results)

def generate_plots(results: List[Dict[str, Any]], output_dir: str):
    """Generate and save the comparative plots."""
    
    # 1. Exploitability vs Iterations
    plt.figure(figsize=(12, 8))
    for res in results:
        if res['history']:
            iters, expls = zip(*res['history'])
            plt.plot(iters, expls, label=res['name'], linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Exploitability (Log Scale)', fontsize=12)
    plt.title('Exploitability vs. Iterations across Solvers', fontsize=14)
    plt.legend(fontsize=10)
    
    # Improve grid and ticks for better detail
    plt.yscale('log')
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    
    # Add more ticks
    import matplotlib.ticker as ticker
    ax = plt.gca()
    # Y-axis: Log scale detail
    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=15))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter()) # readable numbers
    
    # X-axis: More frequency
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    plot_path = os.path.join(output_dir, "exploitability_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Exploitability vs Iterations graph saved to: {plot_path}")
    
    # 1b. Exploitability vs Iterations (Linear Scale - Detailed)
    # Filter out Random Solver to see better detail for converging solvers
    plt.figure(figsize=(12, 8))
    for res in results:
        if res['name'] != "Random Solver" and res['history']:
            iters, expls = zip(*res['history'])
            plt.plot(iters, expls, label=res['name'], linewidth=2)
            
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Exploitability (Linear Scale)', fontsize=12)
    plt.title('Exploitability vs. Iterations (Detailed View)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Detailed ticks
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(20))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    plot_path_linear = os.path.join(output_dir, "exploitability_comparison_linear.png")
    plt.tight_layout()
    plt.savefig(plot_path_linear, dpi=300)
    print(f"Detailed linear graph saved to: {plot_path_linear}")
    
    # 2. Exploitability vs Memory Usage (Scatter Plot of Final Values)
    plt.figure(figsize=(12, 8))
    
    for res in results:
        plt.scatter(res['final_memory_mb'], res['final_exploitability'], s=100, label=res['name'])
        plt.annotate(res['name'], (res['final_memory_mb'], res['final_exploitability']), 
                     xytext=(5, 5), textcoords='offset points')
        
    plt.xlabel('Memory Usage (MB)', fontsize=12)
    plt.ylabel('Final Exploitability (Log Scale)', fontsize=12)
    plt.title('Exploitability vs. Memory Usage', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.yscale('log')
    
    mem_plot_path = os.path.join(output_dir, "exploitability_vs_memory.png")
    plt.tight_layout()
    plt.savefig(mem_plot_path, dpi=300)
    print(f"Exploitability vs Memory graph saved to: {mem_plot_path}")


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary table of the results."""
    print(f"\n{'='*90}")
    print(f"  Results Summary")
    print(f"{'='*90}")
    print(f"  {'Algorithm':<20} {'Final Exploitability':>25} {'Final Memory (MB)':>20}")
    print(f"  {'-'*20} {'-'*25} {'-'*20}")
    
    # Sort by performance (lower exploitability is better)
    sorted_results = sorted(results, key=lambda x: x['final_exploitability'])
    
    for res in sorted_results:
        print(f"  {res['name']:<20} {res['final_exploitability']:>25.6f} {res['final_memory_mb']:>20.6f}")
    print(f"{'='*90}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFR Analysis.")
    parser.add_argument("-T", "--iterations", type=int, default=10000, help="Total iterations (T)")
    parser.add_argument("-t", "--interval", type=int, default=100, help="Calculation interval (t)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    run_analysis(args.iterations, args.interval, args.seed, args.output)
