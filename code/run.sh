#!/bin/bash

# Compile the code
make clean
make

# Run all experiments
echo "Running all experiments..."
./mnist_optimiser.out mnist_data -run_experiments

# Analyze results
echo "Analyzing results..."
python3 analyze_results.py

echo "Done!" 