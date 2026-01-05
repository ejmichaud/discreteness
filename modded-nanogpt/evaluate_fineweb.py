#!/usr/bin/env python3
"""
Evaluate model on FineWeb documents (training distribution).

Usage:
    python evaluate_fineweb.py --checkpoint model.pt --num_docs 100
    python evaluate_fineweb.py --checkpoint model.pt --num_docs 1000 --max_tokens 1024
"""

import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

from inference import load_checkpoint, compute_token_losses


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on FineWeb")
    parser.add_argument("--checkpoint", type=str, default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--num_docs", type=int, default=100, help="Number of documents to evaluate")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per document (truncate longer docs)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, mps")
    parser.add_argument("--dataset", type=str, default="sample-10BT", 
                        help="FineWeb dataset variant: sample-10BT, sample-100BT")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N documents")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    if device.type == "cpu":
        model = model.float()
    model.eval()
    
    # Load dataset (streaming)
    print(f"Loading FineWeb dataset ({args.dataset}, streaming)...")
    from datasets import load_dataset
    fw = load_dataset("HuggingFaceFW/fineweb", name=args.dataset, split="train", streaming=True)
    
    # Skip if requested
    if args.skip > 0:
        print(f"Skipping first {args.skip} documents...")
        fw = fw.skip(args.skip)
    
    # Evaluate
    print(f"Evaluating {args.num_docs} documents (max {args.max_tokens} tokens each)...")
    
    losses = []
    token_counts = []
    total_tokens = 0
    weighted_loss_sum = 0
    
    start_time = time.time()
    
    for doc in tqdm(fw.take(args.num_docs), total=args.num_docs):
        text = doc["text"]
        
        try:
            result = compute_token_losses(model, text)
        except Exception as e:
            print(f"Error processing document: {e}")
            continue
        
        if len(result.losses) < 1:
            continue
        
        # Truncate if needed
        n_tokens = min(len(result.losses), args.max_tokens)
        doc_losses = result.losses[:n_tokens]
        
        # Per-document stats
        doc_loss = doc_losses.mean().item()
        losses.append(doc_loss)
        token_counts.append(n_tokens)
        
        # For weighted average (by token count)
        total_tokens += n_tokens
        weighted_loss_sum += doc_losses.sum().item()
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    weighted_mean_loss = weighted_loss_sum / total_tokens if total_tokens > 0 else 0
    
    print()
    print("=" * 60)
    print(f"FineWeb Evaluation Results")
    print("=" * 60)
    print(f"Documents evaluated:     {len(losses)}")
    print(f"Total tokens:            {total_tokens:,}")
    print(f"Avg tokens/doc:          {np.mean(token_counts):.0f}")
    print(f"Time elapsed:            {elapsed:.1f}s ({elapsed/len(losses)*1000:.0f}ms/doc)")
    print()
    print(f"Per-document mean loss:  {mean_loss:.4f} nats (std: {std_loss:.4f})")
    print(f"Per-document perplexity: {np.exp(mean_loss):.2f}")
    print()
    print(f"Token-weighted loss:     {weighted_mean_loss:.4f} nats")
    print(f"Token-weighted PPL:      {np.exp(weighted_mean_loss):.2f}")
    print("=" * 60)
    
    # Show some percentiles
    percentiles = [10, 25, 50, 75, 90]
    print("\nLoss percentiles:")
    for p in percentiles:
        val = np.percentile(losses, p)
        print(f"  {p}th: {val:.3f} nats (PPL: {np.exp(val):.1f})")


if __name__ == "__main__":
    main()
    # Force exit to avoid hanging from datasets background threads
    import os
    os._exit(0)

