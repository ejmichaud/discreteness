"""
Minimal evaluation tracker: loads documents, prepares batched inputs, stores losses.
All eval documents are packed into a single forward pass (like training).

Usage:
    tracker = EvalTracker("eval_docs/", eval_steps=[0, 100, 500, 1000, 1800])
    
    # In training loop:
    if tracker.should_eval(step):
        model.eval()
        with torch.no_grad():
            inp, tgt, seqlens = tracker.get_batch()
            losses = model(inp, tgt, seqlens, fwd_cfg, per_token=True)
            tracker.record(step, losses)
        model.train()
    
    # After training:
    tracker.save(f"losses_{run_id}.pt")
"""

import os
import torch
from torch import Tensor
import tiktoken

TOKENIZER = tiktoken.get_encoding("gpt2")
BOS_ID = 50256


class EvalTracker:
    """Loads eval documents, prepares batched inputs, stores per-token losses."""
    
    def __init__(
        self,
        doc_dir: str,
        eval_steps: list[int],
        max_tokens: int = 2048,
        device: str = "cuda",
    ):
        self.eval_steps = set(eval_steps)
        self.device = device
        
        # Load and tokenize all .txt files from directory
        self.doc_names = []
        doc_tokens = []
        
        txt_files = sorted(f for f in os.listdir(doc_dir) if f.endswith('.txt'))
        for fname in txt_files:
            with open(os.path.join(doc_dir, fname), 'r') as f:
                text = f.read()
            
            tokens = [BOS_ID] + TOKENIZER.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                print(f"Warning: truncated {fname} to {len(tokens)} tokens")
            
            self.doc_names.append(fname)
            doc_tokens.append(tokens)
        
        # Build packed batch (single forward pass for all docs)
        # Same varlen format as training: concatenate all docs, track boundaries with seqlens
        all_tokens = []
        self.doc_lengths = []  # Length of each doc (for splitting losses later)
        cum_len = 0
        seqlens = [0]
        
        for tokens in doc_tokens:
            all_tokens.extend(tokens)
            doc_len = len(tokens) - 1  # Number of predictions (input/target pairs)
            self.doc_lengths.append(doc_len)
            cum_len += doc_len
            seqlens.append(cum_len)
        
        all_tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
        
        # Build input/target by removing last token of each doc from input
        # and first token of each doc from target
        inp_parts = []
        tgt_parts = []
        offset = 0
        for tokens in doc_tokens:
            n = len(tokens)
            inp_parts.append(all_tokens[offset:offset + n - 1])
            tgt_parts.append(all_tokens[offset + 1:offset + n])
            offset += n
        
        self.inp = torch.cat(inp_parts)
        self.tgt = torch.cat(tgt_parts)
        self.seqlens = torch.tensor(seqlens, dtype=torch.int32, device=device)
        
        # Results: step -> list of per-doc loss tensors
        self.results: dict[int, list[Tensor]] = {}
        
        print(f"EvalTracker: {len(self.doc_names)} docs from {doc_dir}")
        print(f"  Files: {self.doc_names}")
        print(f"  Token counts: {self.doc_lengths}")
        print(f"  Total tokens: {len(self.inp)}")
        print(f"  Eval steps: {len(self.eval_steps)} steps")
    
    def should_eval(self, step: int) -> bool:
        return step in self.eval_steps
    
    def get_batch(self) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (input_seq, target_seq, seqlens) for single batched forward pass."""
        return self.inp, self.tgt, self.seqlens
    
    def record(self, step: int, losses: Tensor):
        """Split and store per-token losses by document."""
        losses = losses.cpu()
        doc_losses = torch.split(losses, self.doc_lengths)
        self.results[step] = [l.half() for l in doc_losses]
    
    def save(self, path: str):
        """Save results to disk."""
        torch.save({
            'results': self.results,
            'doc_names': self.doc_names,
            'doc_lengths': self.doc_lengths,
            'eval_steps': sorted(self.eval_steps),
        }, path)
        print(f"Saved eval results to {path}")
    
    @staticmethod
    def load(path: str) -> dict:
        """Load results from disk."""
        return torch.load(path)
