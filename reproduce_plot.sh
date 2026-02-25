#!/bin/bash
# Script to reproduce SparseGPT vs Magnitude pruning comparison

# Model to use (change to facebook/opt-175b if you have access)
# MODEL="facebook/opt-125m"
MODEL="facebook/opt-350m"
DATASET="c4"
OUTPUT_DIR="results"

mkdir -p $OUTPUT_DIR

# Sparsity levels from the plot
SPARSITIES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

echo "Running experiments for model: $MODEL"

# Run dense baseline
echo "Running dense baseline..."
python opt.py $MODEL $DATASET > $OUTPUT_DIR/dense.log 2>&1

# Run SparseGPT at different sparsities
for sparsity in "${SPARSITIES[@]}"; do
    if [ "$sparsity" != "0.0" ]; then
        echo "Running SparseGPT with sparsity=$sparsity..."
        python opt.py $MODEL $DATASET --sparsity $sparsity > $OUTPUT_DIR/sparsegpt_${sparsity}.log 2>&1
    fi
done

# Run Magnitude pruning at different sparsities
for sparsity in "${SPARSITIES[@]}"; do
    if [ "$sparsity" != "0.0" ]; then
        echo "Running Magnitude pruning with sparsity=$sparsity..."
        python opt.py $MODEL $DATASET --sparsity $sparsity --gmp > $OUTPUT_DIR/magnitude_${sparsity}.log 2>&1
    fi
done

echo "All experiments complete. Results in $OUTPUT_DIR/"
