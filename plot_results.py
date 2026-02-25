#!/usr/bin/env python3
"""
Parse SparseGPT experiment results and reproduce the sparsity vs perplexity plot.
"""

import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def parse_perplexity_from_log(log_file, dataset='wikitext2'):
    """
    Parse perplexity from a log file.
    
    Args:
        log_file: Path to log file
        dataset: Which dataset perplexity to extract
        
    Returns:
        Perplexity value or None if not found
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Look for the pattern "wikitext2" followed by "Perplexity: X.XXX"
        pattern = f"{dataset}.*?Perplexity: ([\d.]+)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
    
    return None


def collect_results(results_dir='results', dataset='wikitext2'):
    """
    Collect all experiment results.
    
    Returns:
        Dictionary with method -> [(sparsity, perplexity), ...]
    """
    results = {
        'dense': [],
        'sparsegpt': [],
        'magnitude': []
    }
    
    # Parse dense baseline
    dense_log = os.path.join(results_dir, 'dense.log')
    if os.path.exists(dense_log):
        ppl = parse_perplexity_from_log(dense_log, dataset)
        if ppl:
            results['dense'].append((0.0, ppl))
    
    # Parse SparseGPT results
    for log_file in glob.glob(os.path.join(results_dir, 'sparsegpt_*.log')):
        sparsity = float(re.search(r'sparsegpt_([\d.]+)\.log', log_file).group(1))
        ppl = parse_perplexity_from_log(log_file, dataset)
        if ppl:
            results['sparsegpt'].append((sparsity, ppl))
    
    # Parse Magnitude pruning results
    for log_file in glob.glob(os.path.join(results_dir, 'magnitude_*.log')):
        sparsity = float(re.search(r'magnitude_([\d.]+)\.log', log_file).group(1))
        ppl = parse_perplexity_from_log(log_file, dataset)
        if ppl:
            results['magnitude'].append((sparsity, ppl))
    
    # Sort by sparsity
    for method in results:
        results[method].sort(key=lambda x: x[0])
    
    return results


def plot_results(results, output_file='sparsity_vs_perplexity.png', model_name='OPT-175B'):
    """
    Create the sparsity vs perplexity comparison plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Dense baseline (horizontal line)
    if results['dense']:
        dense_ppl = results['dense'][0][1]
        ax.axhline(y=dense_ppl, color='gray', linestyle='--', linewidth=2, label='Dense')
    
    # SparseGPT
    if results['sparsegpt']:
        sparsities, ppls = zip(*results['sparsegpt'])
        ax.plot(sparsities, ppls, 'o-', color='#1f77b4', linewidth=2, 
                markersize=8, label='SparseGPT')
    
    # Magnitude pruning
    if results['magnitude']:
        sparsities, ppls = zip(*results['magnitude'])
        ax.plot(sparsities, ppls, 's-', color='#ff7f0e', linewidth=2, 
                markersize=8, label='Magnitude')
    
    ax.set_xlabel('Sparsity', fontsize=12)
    ax.set_ylabel('Perplexity on raw WikiText2', fontsize=12)
    ax.set_title(f'Sparsity-vs-perplexity comparison on {model_name}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()


def print_results_table(results):
    """
    Print results in a table format.
    """
    print("\n" + "="*70)
    print(f"{'Method':<15} {'Sparsity':<12} {'Perplexity':<12}")
    print("="*70)
    
    # Dense
    if results['dense']:
        print(f"{'Dense':<15} {0.0:<12.1f} {results['dense'][0][1]:<12.3f}")
    
    # SparseGPT
    for sparsity, ppl in results['sparsegpt']:
        print(f"{'SparseGPT':<15} {sparsity:<12.1f} {ppl:<12.3f}")
    
    # Magnitude
    for sparsity, ppl in results['magnitude']:
        print(f"{'Magnitude':<15} {sparsity:<12.1f} {ppl:<12.3f}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot SparseGPT results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing result logs')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'],
                        help='Which dataset perplexity to plot')
    parser.add_argument('--output', type=str, default='sparsity_vs_perplexity.png',
                        help='Output plot filename')
    parser.add_argument('--model_name', type=str, default='OPT-125M',
                        help='Model name for plot title')
    
    args = parser.parse_args()
    
    # Collect results
    print(f"Collecting results from {args.results_dir}/...")
    results = collect_results(args.results_dir, args.dataset)
    
    # Print table
    print_results_table(results)
    
    # Create plot
    plot_results(results, args.output, args.model_name)
