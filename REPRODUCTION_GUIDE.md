# Reproducing SparseGPT Figure 1 Results

This guide explains how to reproduce the sparsity-vs-perplexity comparison plot from the SparseGPT paper.

## Quick Start

### Option 1: Using the Bash Script (Recommended for batch runs)

```bash
cd baselines/sparsegpt
chmod +x reproduce_plot.sh

# Edit the script to set your desired MODEL
# Then run:
./reproduce_plot.sh

# Parse results and create plot
python plot_results.py --model_name "OPT-125M"
```

### Option 2: Using the Interactive Notebook

```bash
cd baselines/sparsegpt
jupyter notebook reproduce_results.ipynb
```

Run cells sequentially to execute experiments and visualize results.

### Option 3: Manual Commands

Run individual experiments:

```bash
# Dense baseline
python opt.py facebook/opt-125m c4

# SparseGPT at 50% sparsity
python opt.py facebook/opt-125m c4 --sparsity 0.5

# Magnitude pruning at 50% sparsity  
python opt.py facebook/opt-125m c4 --sparsity 0.5 --gmp
```

## Understanding the Methods

### SparseGPT (Blue line in plot)
- **Method**: Layer-wise Hessian-based pruning
- **Algorithm**: Uses second-order information (Hessian) to determine which weights to prune
- **Implementation**: `sparsegpt.py` - computes inverse Hessian and optimally removes weights
- **Command**: `--sparsity X` (no --gmp flag)

### Magnitude Pruning (Orange line in plot)
- **Method**: Global magnitude-based pruning
- **Algorithm**: Simply removes weights with smallest absolute values
- **Implementation**: In `opt_eval()` function - threshold based on sorted weight magnitudes
- **Command**: `--sparsity X --gmp`

### Dense Baseline (Dotted line in plot)
- **Method**: Original unpruned model
- **Command**: No sparsity flags

## Important Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sparsity` | 0.0 | Target sparsity fraction (0.0-1.0) |
| `--gmp` | False | Use magnitude pruning instead of SparseGPT |
| `--nsamples` | 128 | Number of calibration samples |
| `--percdamp` | 0.01 | Hessian dampening (% of diagonal mean) |
| `--prunen/prunem` | 0 | For N:M structured sparsity |
| `--wbits` | 16 | Quantization bits (for sparse+quant) |

## Model Options

The paper uses OPT-175B, but you can test with smaller models:

| Model | Size | Memory Needed | HuggingFace Name |
|-------|------|---------------|------------------|
| OPT-125M | 125M params | ~1GB | facebook/opt-125m |
| OPT-350M | 350M params | ~2GB | facebook/opt-350m |
| OPT-1.3B | 1.3B params | ~6GB | facebook/opt-1.3b |
| OPT-6.7B | 6.7B params | ~25GB | facebook/opt-6.7b |
| OPT-13B | 13B params | ~50GB | facebook/opt-13b |
| OPT-175B | 175B params | ~700GB+ | Requires Meta access |

## Expected Results

At high sparsity levels (60-80%), SparseGPT should show:
- **Lower perplexity** than magnitude pruning
- **Better preservation** of model quality
- **Smoother degradation** as sparsity increases

Example from paper (OPT-175B):
- At 0% sparsity: ~8.5 perplexity (both methods)
- At 50% sparsity: ~8.5 (SparseGPT) vs ~9.5 (Magnitude)  
- At 80% sparsity: ~10 (SparseGPT) vs ~17 (Magnitude)

## Computational Requirements

### For OPT-175B (original paper):
- Multiple A100 80GB GPUs
- Meta access + checkpoint conversion
- Runtime: Several hours per sparsity level

### For OPT-125M (demo):
- Single GPU (RTX 3090, V100, etc.)
- Public HuggingFace model
- Runtime: ~5-10 minutes per sparsity level

## Datasets

The script evaluates on 3 datasets and reports all perplexities:
- **WikiText2**: Common benchmark, smaller
- **PTB**: Penn TreeBank
- **C4**: Larger, more diverse (used for calibration in paper)

## Tips for Running Experiments

1. **Start small**: Test with opt-125m before scaling up
2. **Use C4 for calibration**: `python opt.py MODEL c4 --sparsity X`
3. **Check logs**: Each run prints perplexity for all 3 eval datasets
4. **Save models**: Use `--save path/to/checkpoint` to save pruned models
5. **Monitor memory**: Use `nvidia-smi` to track GPU usage
6. **Batch experiments**: Run overnight for multiple sparsity levels

## Troubleshooting

### Out of Memory
- Use smaller model (opt-125m instead of opt-1.3b)
- Reduce `--nsamples` (e.g., 64 instead of 128)
- Reduce `--blocksize` (default 128)

### Model Download Issues  
- Ensure `transformers` library is up to date
- Check HuggingFace access token if needed
- Use `cache_dir` parameter if needed

### Perplexity Issues
- Magnitude pruning should show rapid degradation at high sparsity
- SparseGPT should be more stable
- If both methods fail similarly, check model loading

## Citation

```bibtex
@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}
```
