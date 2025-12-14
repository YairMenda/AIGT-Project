#!/usr/bin/env python3
"""
Kuhn Poker CFR Solvers
======================

Train and compare CFR algorithm variants for Kuhn Poker.

Usage:
    python main.py train vanilla -n 10000   # Train Vanilla CFR
    python main.py train cfr+ -n 5000       # Train CFR+
    python main.py compare -n 10000         # Compare all variants
    python main.py info                     # Show information sets
"""

import argparse
from kuhn_poker import KuhnPoker, Card
from kuhn_poker.game import get_all_info_sets, build_game_tree
from kuhn_poker.solvers import (
    VanillaCFR, CFRPlus, DiscountedCFR, LinearCFR,
    ExponentialCFR, SoftmaxCFR
)


def cmd_train(args):
    """Train a CFR solver."""
    solvers = {
        "vanilla": VanillaCFR,
        "cfr+": lambda: CFRPlus(alternating=True),
        "dcfr": lambda: DiscountedCFR(alpha=args.alpha, beta=args.beta, gamma=args.gamma),
        "linear": LinearCFR,
        "exponential": lambda: ExponentialCFR(rate=0.001),
        "softmax": lambda: SoftmaxCFR(temperature=0.1),
    }
    
    if args.solver not in solvers:
        available = ", ".join(solvers.keys())
        print(f"Unknown solver: {args.solver}. Available: {available}")
        return
    
    print(f"\n{'='*60}")
    print(f"  Training {args.solver.upper()} Solver")
    print(f"  Iterations: {args.iterations}")
    print(f"{'='*60}\n")
    
    # Create solver
    solver_class = solvers[args.solver]
    if callable(solver_class) and not isinstance(solver_class, type):
        solver = solver_class()
    else:
        solver = solver_class(seed=args.seed)
    
    # Training callback
    def callback(iteration, exploitability):
        print(f"  Iteration {iteration:6d}: Exploitability = {exploitability:.6f}")
    
    # Train
    solver.train(
        args.iterations, 
        callback=callback if args.verbose else None,
        compute_exploitability_every=args.exploitability_every
    )
    
    # Print results
    solver.print_strategy()
    
    # Save if requested
    if args.output:
        solver.save_strategy(args.output)
        print(f"Strategy saved to: {args.output}")
    
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Algorithm: {solver.name}")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Final Exploitability: {solver.compute_exploitability():.6f}")
    print(f"{'='*60}\n")


def cmd_compare(args):
    """Compare all CFR solver variants."""
    print(f"\n{'='*70}")
    print(f"  Comparative Analysis of CFR Algorithm Variants")
    print(f"  Iterations per solver: {args.iterations}")
    print(f"{'='*70}\n")
    
    # Create all solvers
    solvers = [
        ("Vanilla CFR", VanillaCFR(seed=args.seed)),
        ("CFR+", CFRPlus(alternating=True, seed=args.seed)),
        ("DCFR", DiscountedCFR(alpha=1.5, beta=0, gamma=2, seed=args.seed)),
        ("Linear CFR", LinearCFR(seed=args.seed)),
        ("Exponential", ExponentialCFR(rate=0.001, seed=args.seed)),
        ("Softmax", SoftmaxCFR(temperature=0.1, seed=args.seed)),
    ]
    
    results = []
    
    for name, solver in solvers:
        print(f"\nTraining {name}...")
        
        # Track exploitability over time
        exploitabilities = []
        
        def callback(iteration, expl):
            exploitabilities.append((iteration, expl))
            if args.verbose:
                print(f"  Iter {iteration}: {expl:.6f}")
        
        solver.train(
            args.iterations,
            callback=callback,
            compute_exploitability_every=args.exploitability_every
        )
        
        final_expl = solver.compute_exploitability()
        
        results.append({
            "name": name,
            "solver": solver,
            "final_exploitability": final_expl,
            "exploitability_history": exploitabilities,
        })
        
        print(f"  Final exploitability: {final_expl:.6f}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  Results Summary")
    print(f"{'='*70}")
    print(f"\n  {'Algorithm':<20} {'Final Exploitability':>25} {'Convergence':>15}")
    print(f"  {'-'*20} {'-'*25} {'-'*15}")
    
    for r in sorted(results, key=lambda x: x['final_exploitability']):
        convergence = "Excellent" if r['final_exploitability'] < 0.01 else \
                     "Good" if r['final_exploitability'] < 0.05 else "Converging"
        print(f"  {r['name']:<20} {r['final_exploitability']:>25.6f} {convergence:>15}")
    
    # Print strategy comparison for key information sets
    print(f"\n{'='*70}")
    print(f"  Strategy Comparison at Key Decision Points")
    print(f"{'='*70}\n")
    
    key_info_sets = ["J", "Jcb", "Q", "Qcb", "K", "Kb"]
    
    for info_set in key_info_sets:
        print(f"\n  Information Set: '{info_set}'")
        for r in results:
            strategy = r["solver"].get_average_strategy()
            if info_set in strategy:
                probs = strategy[info_set]
                data = r["solver"].info_sets[info_set]
                prob_str = ", ".join(f"{a}: {p:.3f}" for a, p in zip(data.actions, probs))
                print(f"    {r['name']:<15}: [{prob_str}]")
    
    print(f"\n{'='*70}")
    print(f"  Comparison Complete")
    print(f"{'='*70}\n")
    
    # Save comparison plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            for r in results:
                if r['exploitability_history']:
                    iters, expls = zip(*r['exploitability_history'])
                    plt.plot(iters, expls, label=r['name'], linewidth=2)
            
            plt.xlabel('Iterations', fontsize=12)
            plt.ylabel('Exploitability', fontsize=12)
            plt.title('CFR Algorithm Convergence Comparison', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"  Convergence plot saved to: {args.plot}")
            
        except ImportError:
            print("  (matplotlib not available for plotting)")
    
    return results


def cmd_info(args):
    """Display all information sets."""
    game = KuhnPoker()
    info_sets = get_all_info_sets(game)
    
    print(f"\n{'='*50}")
    print(f"  Kuhn Poker Information Sets")
    print(f"{'='*50}\n")
    
    print("Format: InfoSet -> [Legal Actions]")
    print("InfoSet = <Card><ActionHistory>")
    
    # Group by card
    for card in ['J', 'Q', 'K']:
        print(f"\n  {card} (Jack/Queen/King):")
        for info_set, actions in sorted(info_sets.items()):
            if info_set.startswith(card):
                actions_str = ", ".join(str(a) for a in actions)
                history = info_set[1:] if len(info_set) > 1 else "(start)"
                print(f"    '{info_set}' [history: {history}] -> [{actions_str}]")
    
    print(f"\n{'='*50}")
    print(f"  Total Information Sets: {len(info_sets)}")
    print(f"{'='*50}\n")


def cmd_tree(args):
    """Display the game tree structure."""
    game = KuhnPoker()
    
    print(f"\n{'='*60}")
    print(f"  Kuhn Poker Game Tree")
    print(f"{'='*60}\n")
    
    print("Game tree for deal: P0=Jack, P1=Queen\n")
    
    root = build_game_tree(game, (Card.JACK, Card.QUEEN))
    
    def print_tree(node, indent=0, action=None):
        prefix = "  " * indent
        
        if action:
            print(f"{prefix}└─ {action.name}")
            prefix = "  " * (indent + 1)
        
        state = node.state
        
        if node.is_terminal:
            winner_card = state.cards[state.winner]
            print(f"{prefix}[Terminal] Winner: P{state.winner} ({winner_card}), Payoffs: {state.payoffs}")
        else:
            player = state.current_player
            card = state.cards[player]
            info_set = state.get_info_set(player)
            print(f"{prefix}[P{player} to act | Card: {card} | InfoSet: '{info_set}']")
            
            for action, child in node.children.items():
                print_tree(child, indent + 1, action)
    
    print_tree(root)
    
    print(f"\n{'='*60}")
    print("Note: The tree structure is the same for all deals,")
    print("but the outcomes depend on which player has the higher card.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Kuhn Poker CFR Solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train vanilla -n 10000    # Train Vanilla CFR
  python main.py train cfr+ -n 5000 -v     # Train CFR+ with verbose output
  python main.py train dcfr -n 10000       # Train Discounted CFR
  python main.py compare -n 10000          # Compare all CFR variants
  python main.py info                      # Show information sets
  python main.py tree                      # Show game tree
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a CFR solver")
    train_parser.add_argument("solver", choices=["vanilla", "cfr+", "dcfr", "linear", "exponential", "softmax"],
                             help="CFR solver variant to train")
    train_parser.add_argument("-n", "--iterations", type=int, default=10000,
                             help="Number of training iterations")
    train_parser.add_argument("-o", "--output", type=str, default=None,
                             help="Save trained strategy to file")
    train_parser.add_argument("-v", "--verbose", action="store_true",
                             help="Show training progress")
    train_parser.add_argument("-s", "--seed", type=int, default=None,
                             help="Random seed")
    train_parser.add_argument("--exploitability-every", type=int, default=100,
                             help="Compute exploitability every N iterations")
    train_parser.add_argument("--alpha", type=float, default=1.5,
                             help="DCFR alpha parameter")
    train_parser.add_argument("--beta", type=float, default=0.0,
                             help="DCFR beta parameter")
    train_parser.add_argument("--gamma", type=float, default=2.0,
                             help="DCFR gamma parameter")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare all CFR variants")
    compare_parser.add_argument("-n", "--iterations", type=int, default=10000,
                               help="Number of training iterations per solver")
    compare_parser.add_argument("-s", "--seed", type=int, default=42,
                               help="Random seed for reproducibility")
    compare_parser.add_argument("-v", "--verbose", action="store_true",
                               help="Show detailed training progress")
    compare_parser.add_argument("--exploitability-every", type=int, default=100,
                               help="Compute exploitability every N iterations")
    compare_parser.add_argument("--plot", type=str, default=None,
                               help="Save convergence plot to file")
    
    # Info command
    subparsers.add_parser("info", help="Display all information sets")
    
    # Tree command
    subparsers.add_parser("tree", help="Display the game tree")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    commands = {
        "train": cmd_train,
        "compare": cmd_compare,
        "info": cmd_info,
        "tree": cmd_tree,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
