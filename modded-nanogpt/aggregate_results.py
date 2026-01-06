#!/usr/bin/env python3
"""
Aggregate evaluation results from multi-seed runs.

Usage:
    python aggregate_results.py                    # Process all seeds in logs/
    python aggregate_results.py --output results.pt  # Custom output filename
"""

import argparse
import glob
import os
import torch

def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed evaluation results")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing log files")
    parser.add_argument("--output", default="aggregated_results.pt", help="Output filename")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 299], help="Seed range (start end)")
    args = parser.parse_args()
    
    start_seed, end_seed = args.seeds
    
    # Find all eval_losses files
    pattern = os.path.join(args.logs_dir, "seed*_eval_losses.pt")
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} result files in {args.logs_dir}/")
    
    if not files:
        print("No results found. Run training first.")
        return
    
    # Load and aggregate results
    all_results = {}
    seeds_found = []
    
    for fpath in files:
        # Extract seed from filename (seed{N}_{uuid}_eval_losses.pt)
        fname = os.path.basename(fpath)
        seed = int(fname.split("_")[0].replace("seed", ""))
        
        if seed < start_seed or seed > end_seed:
            continue
            
        seeds_found.append(seed)
        data = torch.load(fpath, weights_only=False)
        all_results[seed] = data
    
    seeds_found.sort()
    
    # Extract common metadata from first result
    first_data = all_results[seeds_found[0]]
    doc_names = first_data['doc_names']
    doc_lengths = first_data['doc_lengths']
    eval_steps = first_data['eval_steps']
    
    print(f"\nSeeds loaded: {len(seeds_found)} (range {min(seeds_found)}-{max(seeds_found)})")
    print(f"Documents: {doc_names}")
    print(f"Eval steps: {len(eval_steps)} steps")
    
    # Check for missing seeds
    expected = set(range(start_seed, end_seed + 1))
    found = set(seeds_found)
    missing = sorted(expected - found)
    if missing:
        print(f"\nMissing seeds ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    
    # Aggregate: create tensors of shape [num_seeds, num_steps, num_docs, max_doc_len]
    # Store per-doc losses for each seed/step
    aggregated = {
        'seeds': seeds_found,
        'doc_names': doc_names,
        'doc_lengths': doc_lengths,
        'eval_steps': eval_steps,
        'results': all_results,  # seed -> {results, doc_names, ...}
    }
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    
    # Mean loss per document per step across seeds
    summary = {}
    for step in eval_steps:
        step_losses = []
        for seed in seeds_found:
            doc_losses = all_results[seed]['results'].get(step, [])
            if doc_losses:
                # Mean loss per doc at this step for this seed
                means = [l.float().mean().item() for l in doc_losses]
                step_losses.append(means)
        
        if step_losses:
            step_tensor = torch.tensor(step_losses)  # [num_seeds, num_docs]
            summary[step] = {
                'mean': step_tensor.mean(dim=0).tolist(),  # [num_docs]
                'std': step_tensor.std(dim=0).tolist(),
                'per_seed': step_tensor.tolist(),
            }
    
    aggregated['summary'] = summary
    
    # Save
    output_path = os.path.join(args.logs_dir, args.output)
    torch.save(aggregated, output_path)
    print(f"\nSaved aggregated results to {output_path}")
    
    # Print final step summary
    if eval_steps:
        final_step = max(eval_steps)
        if final_step in summary:
            print(f"\nFinal step ({final_step}) mean losses by document:")
            for i, (name, mean, std) in enumerate(zip(doc_names, summary[final_step]['mean'], summary[final_step]['std'])):
                print(f"  {name}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()

