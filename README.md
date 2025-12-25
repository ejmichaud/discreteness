# discreteness

Multi-seed modded-nanogpt speedrun evaluation project.

## Overview

This project performs modded-nanogpt speedruns across multiple seeds and evaluates models on carefully chosen validation documents at hundreds of training checkpoints.

## Setup Instructions

### Prerequisites

- NVIDIA H100 GPUs (8x recommended for full speedrun)
- CUDA 12.6 or compatible
- Python 3.11+
- Git with SSH or HTTPS access

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/ejmichaud/discreteness.git
cd discreteness
```

#### 2. Install uv (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

#### 3. Set up Python environment

Navigate to the modded-nanogpt directory and create a virtual environment:

```bash
cd modded-nanogpt
uv venv
source .venv/bin/activate
```

#### 4. Install PyTorch and dependencies

**IMPORTANT:** The standard modded-nanogpt instructions recommend PyTorch nightly, but as of December 2025, the nightly builds don't have compatible Flash Attention 3 binaries. Instead, use the stable PyTorch 2.9.1 with CUDA 12.6:

```bash
# Install PyTorch 2.9.1 with CUDA 12.6 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
uv pip install numpy tqdm huggingface-hub setuptools

# Install kernels package for Flash Attention 3
uv pip install kernels
```

#### 5. Download training data

Download the FineWeb dataset (first 900M tokens for the speedrun):

```bash
python data/cached_fineweb10B.py 9
```

Note: For testing, you can download just 100M tokens with `python data/cached_fineweb10B.py 1`

#### 6. Verify installation

Test that everything is working:

```bash
python -c "import torch; from kernels import get_kernel; print('Setup successful!'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see:
```
Setup successful!
PyTorch: 2.9.1+cu126
CUDA available: True
```

### Running the Speedrun

To run the full speedrun on 8 GPUs:

```bash
./run.sh
```

For testing with fewer GPUs (e.g., 2 GPUs):

```bash
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

## Key Differences from Standard Setup

The main deviation from the [official modded-nanogpt installation](https://github.com/KellerJordan/modded-nanogpt#running-the-current-record) is:

1. **Use stable PyTorch 2.9.1 instead of nightly** - The nightly builds currently lack Flash Attention 3 binary compatibility
2. **Use `kernels` package** - This provides pre-built Flash Attention 3 kernels compatible with H100 GPUs
3. **Use uv for faster installation** - Significantly speeds up dependency resolution and installation

## Troubleshooting

### Flash Attention Issues

If you encounter Flash Attention errors:
- Verify CUDA version: `nvidia-smi` (should show CUDA 12.6)
- Verify kernels package: `python -c "from kernels import get_kernel; print('OK')"`
- The kernels package automatically downloads FA3 kernels from HuggingFace on first run

### Out of Memory

If you run out of GPU memory:
- Reduce the number of GPUs: `torchrun --standalone --nproc_per_node=N train_gpt.py` (where N < 8)
- Check that no other processes are using GPU: `nvidia-smi`

### Data Loading Errors

If you see `StopIteration` errors:
- Ensure you downloaded enough data: `python data/cached_fineweb10B.py 9`
- Check data directory: `ls -lh data/fineweb10B/`

## Project Structure

```
discreteness/
├── README.md                    # This file
└── modded-nanogpt/             # Vendored copy of modded-nanogpt speedrun
    ├── train_gpt.py            # Main training script
    ├── train_gpt_medium.py     # GPT-2 Medium track
    ├── data/                   # Dataset download scripts
    ├── run.sh                  # Run script for 8 GPUs
    └── .venv/                  # Python virtual environment
```

## References

- [modded-nanogpt repository](https://github.com/KellerJordan/modded-nanogpt)
- [Flash Attention 3 kernels](https://huggingface.co/varunneal/flash-attention-3)
- [uv package manager](https://github.com/astral-sh/uv)

## License

See individual component licenses in their respective directories.
