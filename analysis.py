#!/usr/bin/env python3
"""
Analysis and Comparison of CFR Solvers for Kuhn Poker.

This script implements the following analysis plan:
1. Run all solvers for a fixed number of iterations (T).
2. Calculate the Exploitability of the strategy profile at intervals of t.
3. Track memory usage (estimated) for each solver.
4. Track CPU time for each solver.
5. Extract and compare strategy profiles against theoretical GTO.
6. Generate comparative graphs plotting:
   - Exploitability (y-axis) vs. Iterations (x-axis)
   - Exploitability (y-axis) vs. Memory Usage (x-axis)
   - Exploitability (y-axis) vs. CPU Time (x-axis)
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
        memory_usage_history = []  # Track memory usage at each interval
        cpu_time_history = []  # Track CPU time at each interval
        start_time = time.perf_counter()

        def callback(iteration, expl):
            current_time = time.perf_counter() - start_time
            exploitability_history.append((iteration, expl))
            cpu_time_history.append((current_time, expl))

            # Estimate memory usage
            mem_usage_bytes = get_memory_usage(solver)
            mem_usage_mb = mem_usage_bytes / (1024 * 1024)
            memory_usage_history.append((iteration, mem_usage_mb))

            # Optional: print progress less frequently to avoid clutter
            if iteration % (interval * 10) == 0:
                print(f"  Iter {iteration}: {expl:.6f} (Mem: {mem_usage_mb:.4f} MB, Time: {current_time:.2f}s)")

        solver.train(
            iterations,
            callback=callback,
            compute_exploitability_every=interval
        )

        total_time = time.perf_counter() - start_time
        final_expl = solver.compute_exploitability()
        final_mem = get_memory_usage(solver) / (1024 * 1024)

        # Extract final strategy for GTO comparison
        final_strategy = solver.get_average_strategy()

        print(f"  Final exploitability: {final_expl:.6f}")
        print(f"  Final memory usage: {final_mem:.6f} MB")
        print(f"  Total CPU time: {total_time:.2f} seconds\n")

        results.append({
            "name": name,
            "final_exploitability": final_expl,
            "final_memory_mb": final_mem,
            "total_cpu_time": total_time,
            "history": exploitability_history,
            "memory_history": memory_usage_history,
            "cpu_time_history": cpu_time_history,
            "final_strategy": final_strategy
        })

    # Generate Comparative Graphs
    generate_plots(results, output_dir)
    print_summary(results)
    print_strategy_comparison(results, output_dir)

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

    # 3. Exploitability vs CPU Time
    plt.figure(figsize=(12, 8))
    for res in results:
        if res['cpu_time_history']:
            times, expls = zip(*res['cpu_time_history'])
            plt.plot(times, expls, label=res['name'], linewidth=2)

    plt.xlabel('CPU Time (seconds)', fontsize=12)
    plt.ylabel('Exploitability (Log Scale)', fontsize=12)
    plt.title('Exploitability vs. CPU Time across Solvers', fontsize=14)
    plt.legend(fontsize=10)
    plt.yscale('log')
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)

    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=15))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    cpu_plot_path = os.path.join(output_dir, "exploitability_vs_cpu_time.png")
    plt.tight_layout()
    plt.savefig(cpu_plot_path, dpi=300)
    print(f"Exploitability vs CPU Time graph saved to: {cpu_plot_path}")

    # 4. Final Exploitability vs CPU Time (Scatter - Efficiency comparison)
    plt.figure(figsize=(12, 8))

    for res in results:
        plt.scatter(res['total_cpu_time'], res['final_exploitability'], s=100, label=res['name'])
        plt.annotate(res['name'], (res['total_cpu_time'], res['final_exploitability']),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Total CPU Time (seconds)', fontsize=12)
    plt.ylabel('Final Exploitability (Log Scale)', fontsize=12)
    plt.title('Solver Efficiency: Final Exploitability vs. Total CPU Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.yscale('log')

    efficiency_plot_path = os.path.join(output_dir, "solver_efficiency.png")
    plt.tight_layout()
    plt.savefig(efficiency_plot_path, dpi=300)
    print(f"Solver Efficiency graph saved to: {efficiency_plot_path}")


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary table of the results."""
    print(f"\n{'='*110}")
    print(f"  Results Summary")
    print(f"{'='*110}")
    print(f"  {'Algorithm':<20} {'Final Exploitability':>22} {'Memory (MB)':>15} {'CPU Time (s)':>15}")
    print(f"  {'-'*20} {'-'*22} {'-'*15} {'-'*15}")

    # Sort by performance (lower exploitability is better)
    sorted_results = sorted(results, key=lambda x: x['final_exploitability'])

    for res in sorted_results:
        print(f"  {res['name']:<20} {res['final_exploitability']:>22.6f} {res['final_memory_mb']:>15.4f} {res['total_cpu_time']:>15.2f}")
    print(f"{'='*110}\n")


def print_strategy_comparison(results: List[Dict[str, Any]], output_dir: str):
    """
    Print and save GTO strategy comparison for all solvers.

    Theoretical GTO for Kuhn Poker (Player 1 first action):
    - Jack: Check ~2/3, Bet ~1/3 (bluff)
    - Queen: Check 100% (always check)
    - King: Check ~1/3 (trap), Bet ~2/3 (value)

    Player 2 facing check:
    - Jack: Check 100%, never bet
    - Queen: Check ~2/3, Bet ~1/3
    - King: Bet 100% (value)

    Player 2 facing bet:
    - Jack: Fold 100%
    - Queen: Call ~1/3, Fold ~2/3
    - King: Call 100%
    """
    # Key information sets for GTO comparison
    key_info_sets = {
        # Player 1 first action (holding card, no history)
        "J": "P1 Jack initial",
        "Q": "P1 Queen initial",
        "K": "P1 King initial",
        # Player 2 after check
        "Jc": "P2 Jack after check",
        "Qc": "P2 Queen after check",
        "Kc": "P2 King after check",
        # Player 2 after bet
        "Jb": "P2 Jack after bet",
        "Qb": "P2 Queen after bet",
        "Kb": "P2 King after bet",
    }

    print(f"\n{'='*120}")
    print(f"  Strategy Profile Comparison (GTO Analysis)")
    print(f"{'='*120}\n")

    # Prepare data for CSV output
    strategy_data = []

    for info_set, description in key_info_sets.items():
        print(f"  {description} (info_set: '{info_set}'):")
        print(f"  {'-'*60}")

        for res in results:
            strategy = res.get('final_strategy', {})
            if info_set in strategy:
                probs = strategy[info_set]
                # Determine action names based on info set
                if info_set in ['J', 'Q', 'K']:  # First action
                    actions = ['check', 'bet']
                elif info_set.endswith('c'):  # After check
                    actions = ['check', 'bet']
                else:  # After bet
                    actions = ['fold', 'call']

                prob_str = ", ".join(f"{a}: {p:.3f}" for a, p in zip(actions, probs))
                print(f"    {res['name']:<20}: [{prob_str}]")

                strategy_data.append({
                    'info_set': info_set,
                    'description': description,
                    'solver': res['name'],
                    'action_1': actions[0],
                    'prob_1': probs[0],
                    'action_2': actions[1],
                    'prob_2': probs[1] if len(probs) > 1 else 0
                })
        print()

    print(f"{'='*120}\n")

    # Save strategy comparison to CSV
    csv_path = os.path.join(output_dir, "strategy_comparison.csv")
    with open(csv_path, 'w') as f:
        f.write("info_set,description,solver,action_1,prob_1,action_2,prob_2\n")
        for row in strategy_data:
            f.write(f"{row['info_set']},{row['description']},{row['solver']},"
                    f"{row['action_1']},{row['prob_1']:.4f},{row['action_2']},{row['prob_2']:.4f}\n")
    print(f"Strategy comparison saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFR Analysis.")
    parser.add_argument("-T", "--iterations", type=int, default=10000, help="Total iterations (T)")
    parser.add_argument("-t", "--interval", type=int, default=100, help="Calculation interval (t)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    run_analysis(args.iterations, args.interval, args.seed, args.output)
