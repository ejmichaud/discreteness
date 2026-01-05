#!/usr/bin/env python3
"""
Standalone inference script for modded-nanogpt models.
Supports CPU, CUDA, and MPS devices. No distributed training dependencies.

Usage:
    python inference.py --checkpoint model.pt --prompt "Hello, world!"
    python inference.py --checkpoint model.pt --prompt "Once upon a time" --max_tokens 200 --temperature 0.8
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Use tiktoken for GPT-2 tokenization
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("gpt2")
except ImportError:
    TOKENIZER = None
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")


# -----------------------------------------------------------------------------
# Model Architecture (simplified for inference)

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class Rotary(nn.Module):
    """Simplified rotary embeddings for inference (no YaRN extension)."""
    def __init__(self, head_dim: int, max_seq_len: int, device: torch.device):
        super().__init__()
        # Base frequency = 1/1024, linearly interpolated
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim // 4, dtype=torch.float32, device=device)
        # half-truncate RoPE
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)
        self.attn_scale = 0.1  # default scale

    def forward(self, x_BTHD: Tensor) -> Tensor:
        """Apply rotary embeddings to input tensor."""
        T = x_BTHD.size(-3)
        cos = self.cos[None, :T, None, :]
        sin = self.sin[None, :T, None, :]
        x1, x2 = x_BTHD.chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1)


class CastedLinear(nn.Linear):
    """Linear layer that casts weights to input dtype (for bfloat16 inference)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        
        # Merged QKVO weights
        self.qkvo_w = nn.Parameter(torch.empty(dim * 4, self.hdim))
        # Sparse gated attention
        self.attn_gate = CastedLinear(12, num_heads)

    def forward(self, x: Tensor, rotary: Rotary, ve: Tensor | None = None, 
                sa_lambdas: Tensor | None = None, key_shift: bool = False):
        B, T, _ = x.shape
        
        # Default sa_lambdas if not provided
        if sa_lambdas is None:
            sa_lambdas = torch.ones(2, device=x.device, dtype=x.dtype)
        
        # Compute Q, K, V
        qkv = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x))
        qkv = qkv.view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)
        
        # QK normalization
        q, k = norm(q), norm(k)
        
        # Apply rotary embeddings
        q = rotary(q)
        k = rotary(k)
        
        # Optional key shift (for induction heads)
        if key_shift and T > 1:
            k_shifted = k.clone()
            k_shifted[:, 1:, :, self.head_dim//4:self.head_dim//2] = k[:, :-1, :, self.head_dim//4:self.head_dim//2]
            k_shifted[:, 1:, :, self.head_dim//4 + self.head_dim//2:] = k[:, :-1, :, self.head_dim//4 + self.head_dim//2:]
            k = k_shifted
        
        # Value embeddings
        if ve is not None:
            v = v + ve.view_as(v)
        
        # Reshape for attention: (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=rotary.attn_scale)
        
        # Reshape back: (B, T, num_heads, head_dim)
        y = y.transpose(1, 2)
        
        # Apply attention gate
        gate = torch.sigmoid(self.attn_gate(x[..., :12]))
        y = y * gate.unsqueeze(-1)
        
        # Output projection
        y = y.contiguous().view(B, T, self.hdim)
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))
        
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(hdim, dim))
        self.c_proj = nn.Parameter(torch.empty(hdim, dim))

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.type_as(x))
        x = F.relu(x).square()  # ReGLU^2 activation
        x = F.linear(x, self.c_proj.T.type_as(x))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # Skip attention on layer 6
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx != 6 else None
        self.mlp = MLP(dim)
        self.layer_idx = layer_idx

    def forward(self, x: Tensor, rotary: Rotary, ve: Tensor | None = None,
                sa_lambdas: Tensor | None = None, key_shift: bool = False):
        if self.attn is not None:
            x = x + self.attn(norm(x), rotary, ve, sa_lambdas, key_shift)
        x = x + self.mlp(norm(x))
        return x


class GPTInference(nn.Module):
    """GPT model for inference - simplified from training version."""
    
    def __init__(self, vocab_size: int = 50257, num_layers: int = 11, num_heads: int = 6,
                 head_dim: int = 128, model_dim: int = 768, max_seq_len: int = 8192,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)  # Pad to multiple of 128
        
        # Smear gate
        self.smear_gate = CastedLinear(12, 1)
        
        # Value embeddings (3 sets)
        self.value_embeds = nn.ModuleList([
            nn.Embedding(self.vocab_size, model_dim) for _ in range(3)
        ])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)
        ])
        
        # Rotary embeddings
        self.rotary = Rotary(head_dim, max_seq_len, device)
        
        # LM head (weight-tied with embedding)
        self.lm_head = CastedLinear(model_dim, self.vocab_size)
        
        # Learnable scalars
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        # scalars: [resid_lambdas (11), sa_lambdas (22), smear_lambda (1), backout_lambda (1), skip_lambda (1), padding (4)]
        # Padding is added to make the tensor size divisible by world_size during training
        self.scalars = nn.Parameter(torch.zeros(40))  # Match checkpoint size
        
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: (batch_size, seq_len) tensor of token ids
            
        Returns:
            logits: (batch_size, seq_len, vocab_size) tensor
        """
        B, T = input_ids.shape
        
        # Extract scalars
        resid_lambdas = self.scalars[:self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[self.num_layers:3 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[3 * self.num_layers]
        backout_lambda = self.scalars[3 * self.num_layers + 1]
        skip_lambda = self.scalars[3 * self.num_layers + 2]
        
        # Key shift config (layers with long windows)
        key_shift_layers = {3, 10}
        
        # Skip connections config
        skip_in = [3]
        skip_out = [6]
        skip_connections = []
        backout_layer = 7
        x_backout = None
        
        # Token embedding (weight-tied with lm_head)
        x = F.embedding(input_ids, self.lm_head.weight)
        
        # Value embeddings
        ve_list = [ve(input_ids) for ve in self.value_embeds]
        # Pattern: .12 ... 012
        ve = [ve_list[1], ve_list[2]] + [None] * (self.num_layers - 5) + [ve_list[0], ve_list[1], ve_list[2]]
        
        # Smear token embedding forward
        if T > 1:
            smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :12]))
            x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)
        
        x = x0 = norm(x)
        
        # Forward through blocks
        for i, block in enumerate(self.blocks):
            # Skip connection output
            if i in skip_out:
                gate = torch.sigmoid(skip_lambda)
                x = x + gate * skip_connections.pop()
            
            # Residual scaling
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0
            
            # Block forward
            x = block(x, self.rotary, ve[i], sa_lambdas[i], key_shift=(i in key_shift_layers))
            
            # Skip connection input
            if i in skip_in:
                skip_connections.append(x)
            
            # Store for backout
            if i == backout_layer:
                x_backout = x
        
        # Back out early layer contributions
        if x_backout is not None:
            x = x - backout_lambda * x_backout
        
        x = norm(x)
        
        # LM head with softcapping
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits / 7.5)
        
        return logits

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, 
                 repetition_penalty: float = 1.0,
                 stop_on_eos: bool = True) -> Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: (batch_size, seq_len) tensor of prompt token ids (should include BOS)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = neutral, <1 = more deterministic)
            top_k: if set, only sample from top k tokens
            top_p: if set, nucleus sampling threshold
            repetition_penalty: penalty for repeating tokens (1.0 = no penalty, >1 = discourage)
            stop_on_eos: whether to stop generation when EOS token is produced
            
        Returns:
            (batch_size, seq_len + generated_tokens) tensor of token ids
        """
        eos_token_id = 50256  # GPT-2's <|endoftext|>
        
        for _ in range(max_new_tokens):
            # Get logits for the last position
            logits = self(input_ids)[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if stop_on_eos and (next_token == eos_token_id).any():
                break
        
        return input_ids


def _get_colormap():
    """Get matplotlib colormap for loss visualization."""
    try:
        import matplotlib.pyplot as plt
        return plt.cm.RdYlGn_r  # Red-Yellow-Green reversed (green=low, red=high)
    except ImportError:
        return None

def _loss_to_color(loss: float, min_loss: float = 0.0, max_loss: float = 10.0, cmap=None) -> tuple[str, str]:
    """
    Convert loss value to background and text colors using matplotlib colormap.
    
    Returns:
        (background_color, text_color) tuple - text color is black or white based on luminance
    """
    # Normalize to 0-1 range
    normalized = max(0.0, min(1.0, (loss - min_loss) / (max_loss - min_loss)))
    
    if cmap is not None:
        # Use matplotlib colormap
        rgba = cmap(normalized)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    else:
        # Fallback: simple green to red
        r = int(255 * normalized)
        g = int(255 * (1 - normalized))
        b = 50
    
    bg_color = f"rgb({r},{g},{b})"
    
    # Calculate relative luminance (per WCAG guidelines)
    # https://www.w3.org/TR/WCAG20/#relativeluminancedef
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Choose black or white text based on background luminance
    text_color = "#000" if luminance > 0.5 else "#fff"
    
    return bg_color, text_color


@dataclass
class TokenLossResult:
    """Result of computing per-token losses on a document."""
    tokens: list[int]           # Token IDs (including BOS)
    token_strings: list[str]    # Decoded token strings
    logits: Tensor              # (seq_len, vocab_size) logits
    probs: Tensor               # (seq_len, vocab_size) probabilities  
    losses: Tensor              # (seq_len-1,) per-token cross-entropy losses
    total_loss: float           # Mean loss over all tokens
    perplexity: float           # exp(total_loss)
    
    def to_html(self, min_loss: float = 0.0, max_loss: float = None, cmap: str = "RdYlGn_r") -> str:
        """
        Generate HTML with tokens colored by loss.
        
        Args:
            min_loss: Loss value for "most predictable" color (green)
            max_loss: Loss value for "most surprising" color (red). 
                      If None, uses max observed loss.
            cmap: Matplotlib colormap name (default "RdYlGn_r" = green to red)
        
        Returns:
            HTML string that can be displayed with IPython.display.HTML()
        """
        if max_loss is None:
            max_loss = float(self.losses.max()) * 1.1  # Add 10% headroom
        
        # Get colormap
        try:
            import matplotlib.pyplot as plt
            colormap = plt.cm.get_cmap(cmap)
        except (ImportError, ValueError):
            colormap = None
        
        # CSS styles
        html_parts = ['''
        <style>
        .token-viz { 
            font-family: 'Consolas', 'Monaco', monospace; 
            font-size: 14px; 
            line-height: 2.4;
            padding: 10px;
            background: #fafafa;
            border-radius: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .token { 
            padding: 2px 3px; 
            margin: 1px;
            border-radius: 3px; 
            border: 1px solid #aaa;
            cursor: pointer;
            display: inline;
            text-shadow: 0 0 2px rgba(128,128,128,0.3);
            font-weight: 500;
        }
        .token:hover {
            border-color: #666;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        }
        .bos-token {
            opacity: 0.4;
            font-style: italic;
            border-style: dashed;
        }
        </style>
        <div class="token-viz">
        ''']
        
        # First token is BOS - show it faded
        bos_str = self.token_strings[0].replace('<', '&lt;').replace('>', '&gt;')
        html_parts.append(f'<span class="token bos-token" title="BOS token">{bos_str}</span>')
        
        # Remaining tokens with colors based on loss
        for i, (token_str, loss) in enumerate(zip(self.token_strings[1:], self.losses.tolist())):
            prob = self.probs[i, self.tokens[i + 1]].item()
            bg_color, text_color = _loss_to_color(loss, min_loss, max_loss, colormap)
            
            # Escape HTML entities
            display_str = token_str.replace('<', '&lt;').replace('>', '&gt;')
            # Handle newlines
            display_str = display_str.replace('\n', '↵<br>')
            
            tooltip = f"Token: {repr(token_str)}&#10;Loss: {loss:.3f} nats&#10;Prob: {prob:.4f} ({prob*100:.2f}%)"
            
            html_parts.append(
                f'<span class="token" style="background-color: {bg_color}; color: {text_color};" '
                f'title="{tooltip}">{display_str}</span>'
            )
        
        html_parts.append('</div>')
        
        return ''.join(html_parts)
    
    def display(self, min_loss: float = 0.0, max_loss: float = None, cmap: str = "RdYlGn_r"):
        """Display colored token visualization in Jupyter notebook."""
        from IPython.display import HTML, display
        display(HTML(self.to_html(min_loss, max_loss, cmap)))
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for easy analysis."""
        import pandas as pd
        # losses[i] is the loss for predicting token[i+1] given tokens[0:i+1]
        df = pd.DataFrame({
            'position': range(1, len(self.tokens)),
            'token_id': self.tokens[1:],
            'token': self.token_strings[1:],
            'loss': self.losses.tolist(),
            'prob': [self.probs[i, self.tokens[i+1]].item() for i in range(len(self.tokens)-1)],
        })
        return df


def compute_token_losses(
    model: "GPTInference",
    text: str,
    add_bos: bool = True,
) -> TokenLossResult:
    """
    Compute per-token losses for a document.
    
    Args:
        model: Loaded GPTInference model
        text: Input text to evaluate
        add_bos: Whether to prepend BOS token (default True)
        
    Returns:
        TokenLossResult with per-token losses and statistics
    """
    # Tokenize
    tokens = encode(text, add_bos=add_bos)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    
    # Get logits
    with torch.no_grad():
        logits = model(input_ids)[0]  # (seq_len, vocab_size)
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute per-token cross-entropy losses
    # Loss for position i is the loss of predicting token[i+1] given tokens[0:i+1]
    target_ids = torch.tensor(tokens[1:], dtype=torch.long, device=logits.device)
    losses = F.cross_entropy(logits[:-1], target_ids, reduction='none')
    
    # Get token strings
    token_strings = [decode([t], skip_bos=False) for t in tokens]
    
    total_loss = losses.mean().item()
    perplexity = math.exp(total_loss)
    
    return TokenLossResult(
        tokens=tokens,
        token_strings=token_strings,
        logits=logits.cpu(),
        probs=probs.cpu(),
        losses=losses.cpu(),
        total_loss=total_loss,
        perplexity=perplexity,
    )


def load_checkpoint(checkpoint_path: str, device: torch.device = torch.device("cpu")) -> GPTInference:
    """
    Load a trained checkpoint into the inference model.
    
    Args:
        checkpoint_path: path to the .pt checkpoint file
        device: device to load the model on
        
    Returns:
        Loaded GPTInference model
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # Remove _orig_mod. prefix from torch.compile
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[new_key] = v
    
    # Create model
    model = GPTInference(device=device)
    
    # Load weights (strict=False to handle yarn buffers that aren't in inference model)
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    
    if missing:
        # Filter out expected missing keys (rotary buffers are recomputed)
        truly_missing = [k for k in missing if not k.startswith("rotary.") and not k.startswith("yarn.")]
        if truly_missing:
            print(f"Warning: Missing keys: {truly_missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")
    
    model.to(device)
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {n_params:,} parameters")
    
    return model


BOS_TOKEN_ID = 50256  # GPT-2's <|endoftext|> token, used as BOS in training


def encode(text: str, add_bos: bool = True) -> list[int]:
    """
    Encode text to token ids using GPT-2 tokenizer.
    
    Args:
        text: Text to encode
        add_bos: Whether to prepend BOS token (default True, as model was trained with BOS)
    """
    if TOKENIZER is None:
        raise RuntimeError("tiktoken not installed. Run: pip install tiktoken")
    tokens = TOKENIZER.encode(text)
    if add_bos:
        tokens = [BOS_TOKEN_ID] + tokens
    return tokens


def decode(tokens: list[int], skip_bos: bool = True) -> str:
    """
    Decode token ids to text using GPT-2 tokenizer.
    
    Args:
        tokens: Token ids to decode
        skip_bos: Whether to skip leading BOS token if present
    """
    if TOKENIZER is None:
        raise RuntimeError("tiktoken not installed. Run: pip install tiktoken")
    if skip_bos and tokens and tokens[0] == BOS_TOKEN_ID:
        tokens = tokens[1:]
    return TOKENIZER.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained modded-nanogpt model")
    parser.add_argument("--checkpoint", type=str, default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, I am", help="Text prompt to complete")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (>1 discourages)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Inference dtype")
    parser.add_argument("--no_bos", action="store_true", help="Don't prepend BOS token (not recommended)")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Convert to desired dtype (float32 is safest for CPU)
    if device.type == "cpu":
        model = model.float()
    elif dtype != torch.float32:
        model = model.to(dtype)
        print(f"Converted model to {args.dtype}")
    
    # Encode prompt (with BOS by default - important for this model!)
    prompt_tokens = encode(args.prompt, add_bos=not args.no_bos)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)} (including BOS)" if not args.no_bos else f"Prompt tokens: {len(prompt_tokens)}")
    print("-" * 50)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
    
    # Decode and print (skip_bos=True will remove the leading BOS)
    generated_tokens = output_ids[0].tolist()
    generated_text = decode(generated_tokens, skip_bos=True)
    
    print(f"\nGenerated text:\n{generated_text}")
    print("-" * 50)
    print(f"Total tokens: {len(generated_tokens) - 1}")  # -1 for BOS


if __name__ == "__main__":
    main()

