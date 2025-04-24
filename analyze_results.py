import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create results directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Function to load result data
def load_results(filename):
    return pd.read_csv(filename)

# Plot accuracy vs. iterations for a single configuration
def plot_accuracy_single(data, title, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(data['iteration'], data['accuracy'], marker='o', linestyle='-', markersize=4)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Plot loss vs. iterations for a single configuration
def plot_loss_single(data, title, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(data['iteration'], data['loss'], marker='o', linestyle='-', markersize=4)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Plot accuracy for multiple configurations
def plot_accuracy_comparison(data_dict, title, output_file):
    plt.figure(figsize=(12, 8))
    for label, data in data_dict.items():
        plt.plot(data['iteration'], data['accuracy'], marker='o', linestyle='-', markersize=3, label=label)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()

# Plot loss for multiple configurations
def plot_loss_comparison(data_dict, title, output_file):
    plt.figure(figsize=(12, 8))
    for label, data in data_dict.items():
        plt.plot(data['iteration'], data['loss'], marker='o', linestyle='-', markersize=3, label=label)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()

# Print summary statistics
def print_summary(data_dict):
    print("=== Performance Summary ===")
    for label, data in data_dict.items():
        final_accuracy = data['accuracy'].iloc[-1]
        final_loss = data['loss'].iloc[-1]
        final_iter = data['iteration'].iloc[-1]
        
        # Calculate mean improvement per 10k iterations
        if len(data) > 1 and final_iter > 10000:
            accuracy_points = data['accuracy'].values
            iter_points = data['iteration'].values
            
            # Find points approximately every 10k iterations
            indices = []
            last_idx = 0
            for i in range(1, len(iter_points)):
                if iter_points[i] - iter_points[last_idx] >= 10000:
                    indices.append(i)
                    last_idx = i
            
            if indices:
                improvements = [accuracy_points[indices[i]] - accuracy_points[indices[i-1]] 
                               for i in range(1, len(indices))]
                mean_improvement = np.mean(improvements) if improvements else 0
            else:
                mean_improvement = 0
        else:
            mean_improvement = 0
            
        print(f"{label}:")
        print(f"  Final Accuracy: {final_accuracy:.6f}")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Mean Improvement per 10k iterations: {mean_improvement:.6f}")
        print()

# Main analysis function
def analyze_results():
    print("Analyzing results...")
    
    # Part I - Basic SGD
    print("\n=== Part I: Stochastic Gradient Descent ===")
    sgd_basic = load_results('results_sgd_0.1_10.csv')
    plot_accuracy_single(sgd_basic, 'SGD Accuracy (LR=0.1, Batch Size=10)', 'plots/part1_sgd_accuracy.png')
    plot_loss_single(sgd_basic, 'SGD Loss (LR=0.1, Batch Size=10)', 'plots/part1_sgd_loss.png')
    
    # Part II-A - Effect of Batch Size
    print("\n=== Part II-A: Effect of Batch Size ===")
    sgd_bs1 = load_results('results_sgd_0.1_1.csv')
    sgd_bs10 = sgd_basic.copy()  # Already loaded
    sgd_bs100 = load_results('results_sgd_0.1_100.csv')
    
    batch_size_data = {
        'Batch Size 1': sgd_bs1,
        'Batch Size 10': sgd_bs10,
        'Batch Size 100': sgd_bs100
    }
    
    plot_accuracy_comparison(batch_size_data, 'Effect of Batch Size on Accuracy (LR=0.1)', 
                             'plots/part2a_batch_size_accuracy.png')
    plot_loss_comparison(batch_size_data, 'Effect of Batch Size on Loss (LR=0.1)', 
                         'plots/part2a_batch_size_loss.png')
    print_summary(batch_size_data)
    
    # Part II-A - Effect of Learning Rate
    print("\n=== Part II-A: Effect of Learning Rate ===")
    sgd_lr001 = load_results('results_sgd_0.01_10.csv')
    sgd_lr0001 = load_results('results_sgd_0.001_10.csv')
    
    learning_rate_data = {
        'LR=0.1': sgd_bs10,
        'LR=0.01': sgd_lr001,
        'LR=0.001': sgd_lr0001
    }
    
    plot_accuracy_comparison(learning_rate_data, 'Effect of Learning Rate on Accuracy (Batch Size=10)', 
                             'plots/part2a_learning_rate_accuracy.png')
    plot_loss_comparison(learning_rate_data, 'Effect of Learning Rate on Loss (Batch Size=10)', 
                         'plots/part2a_learning_rate_loss.png')
    print_summary(learning_rate_data)
    
    # Part II-B - Learning Rate Decay
    print("\n=== Part II-B: Learning Rate Decay ===")
    sgd_lr_decay = load_results('results_sgd_lr_decay_0.1_0.001.csv')
    
    lr_decay_data = {
        'SGD (LR=0.1)': sgd_bs10,
        'SGD with LR Decay (0.1→0.001)': sgd_lr_decay
    }
    
    plot_accuracy_comparison(lr_decay_data, 'Effect of Learning Rate Decay on Accuracy', 
                             'plots/part2b_lr_decay_accuracy.png')
    plot_loss_comparison(lr_decay_data, 'Effect of Learning Rate Decay on Loss', 
                         'plots/part2b_lr_decay_loss.png')
    print_summary(lr_decay_data)
    
    # Part II-C - Momentum
    print("\n=== Part II-C: Momentum ===")
    sgd_momentum = load_results('results_sgd_momentum_0.9.csv')
    
    momentum_data = {
        'SGD (LR=0.01)': sgd_lr001,
        'SGD with Momentum (α=0.9)': sgd_momentum
    }
    
    plot_accuracy_comparison(momentum_data, 'Effect of Momentum on Accuracy', 
                             'plots/part2c_momentum_accuracy.png')
    plot_loss_comparison(momentum_data, 'Effect of Momentum on Loss', 
                         'plots/part2c_momentum_loss.png')
    print_summary(momentum_data)
    
    # Part II-D - Combined Approaches
    print("\n=== Part II-D: Combined Approaches ===")
    sgd_combined = load_results('results_sgd_momentum_lr_decay.csv')
    
    combined_data = {
        'SGD (LR=0.1)': sgd_bs10,
        'SGD with LR Decay': sgd_lr_decay,
        'SGD with Momentum': sgd_momentum,
        'SGD with Momentum & LR Decay': sgd_combined
    }
    
    plot_accuracy_comparison(combined_data, 'Effect of Combined Approaches on Accuracy', 
                             'plots/part2d_combined_accuracy.png')
    plot_loss_comparison(combined_data, 'Effect of Combined Approaches on Loss', 
                         'plots/part2d_combined_loss.png')
    print_summary(combined_data)
    
    # Part III - Adam
    print("\n=== Part III: Adam Optimizer ===")
    adam = load_results('results_adam.csv')
    
    adam_data = {
        'SGD (LR=0.1)': sgd_bs10,
        'SGD with Momentum & LR Decay': sgd_combined,
        'Adam': adam
    }
    
    plot_accuracy_comparison(adam_data, 'Adam vs Other Optimization Methods - Accuracy', 
                             'plots/part3_adam_accuracy.png')
    plot_loss_comparison(adam_data, 'Adam vs Other Optimization Methods - Loss', 
                         'plots/part3_adam_loss.png')
    print_summary(adam_data)
    
    print("Analysis complete. Plots saved to 'plots' directory.")

if __name__ == "__main__":
    analyze_results() 