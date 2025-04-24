#!/bin/bash

# Compile the code
make clean
make

# Run the experiments
./mnist_optimiser.out mnist_data 0.01 10 1 -run_experiments

# Run the analysis if Python is available
if command -v pip3 &> /dev/null; then
    echo "Installing required Python packages..."
    pip3 install pandas matplotlib numpy
    python3 analyze_results.py
elif command -v pip &> /dev/null; then
    echo "Installing required Python packages..."
    pip install pandas matplotlib numpy
    python analyze_results.py
else
    echo "Python pip not found. Please install pandas, matplotlib, and numpy manually and run analyze_results.py to generate plots."
fi

echo "Done! Check the plots directory for visualizations." 