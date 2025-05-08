import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# Set style
plt.style.use('default')

# Create results directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def load_results(pattern):
    """Load all CSV files matching the pattern into a list of dataframes."""
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Extract parameters from filename
        params = os.path.basename(f).replace('.csv', '').split('_')
        df['params'] = '_'.join(params[2:])  # Skip 'results' and method name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_learning_curves(dfs, title, ylabel, filename, log_scale=False):
    """Plot learning curves with proper formatting."""
    plt.figure(figsize=(10, 6))
    
    for df in dfs:
        label = df['params'].iloc[0]
        plt.plot(df['iteration'] / 1000, df['value'], label=label)
    
    plt.xlabel('Iterations (×10³)')
    plt.ylabel(ylabel)
    plt.title(title)
    if log_scale:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_hyperparameter_sweep(df, param_name, title, filename):
    """Plot hyperparameter sweep results."""
    plt.figure(figsize=(10, 6))
    
    # Group by parameter value and calculate mean/std
    grouped = df.groupby(param_name)['value'].agg(['mean', 'std']).reset_index()
    
    plt.errorbar(grouped[param_name], grouped['mean'], yerr=grouped['std'], 
                fmt='o-', capsize=5)
    
    plt.xlabel(f'{param_name} value')
    plt.ylabel('Final accuracy (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_sgd():
    """Analyze basic SGD results."""
    # Load results
    sgd_dfs = load_results('results_sgd_*.csv')
    
    # Plot learning curves
    plot_learning_curves(
        [sgd_dfs[sgd_dfs['params'] == p] for p in sgd_dfs['params'].unique()],
        'SGD Learning Curves',
        'Loss',
        'part1_sgd_loss.png',
        log_scale=True
    )
    
    # Plot accuracy curves
    plot_learning_curves(
        [sgd_dfs[sgd_dfs['params'] == p] for p in sgd_dfs['params'].unique()],
        'SGD Accuracy Curves',
        'Accuracy (%)',
        'part1_sgd_accuracy.png'
    )

def analyze_momentum():
    """Analyze momentum results."""
    # Load results
    momentum_dfs = load_results('results_sgd_momentum_*.csv')
    
    # Plot learning curves
    plot_learning_curves(
        [momentum_dfs[momentum_dfs['params'] == p] for p in momentum_dfs['params'].unique()],
        'SGD with Momentum Learning Curves',
        'Loss',
        'part2c_momentum_loss.png',
        log_scale=True
    )
    
    # Plot accuracy curves
    plot_learning_curves(
        [momentum_dfs[momentum_dfs['params'] == p] for p in momentum_dfs['params'].unique()],
        'SGD with Momentum Accuracy Curves',
        'Accuracy (%)',
        'part2c_momentum_accuracy.png'
    )

def analyze_lr_decay():
    """Analyze learning rate decay results."""
    # Load results
    lr_decay_dfs = load_results('results_sgd_lr_decay_*.csv')
    
    # Plot learning curves
    plot_learning_curves(
        [lr_decay_dfs[lr_decay_dfs['params'] == p] for p in lr_decay_dfs['params'].unique()],
        'Learning Rate Decay Curves',
        'Loss',
        'part2b_lr_decay_loss.png',
        log_scale=True
    )
    
    # Plot accuracy curves
    plot_learning_curves(
        [lr_decay_dfs[lr_decay_dfs['params'] == p] for p in lr_decay_dfs['params'].unique()],
        'Learning Rate Decay Accuracy Curves',
        'Accuracy (%)',
        'part2b_lr_decay_accuracy.png'
    )

def analyze_combined():
    """Analyze combined momentum and learning rate decay results."""
    # Load results
    combined_dfs = load_results('results_sgd_momentum_lr_decay_*.csv')
    
    # Plot learning curves
    plot_learning_curves(
        [combined_dfs[combined_dfs['params'] == p] for p in combined_dfs['params'].unique()],
        'Combined Momentum and Learning Rate Decay',
        'Loss',
        'part2d_combined_loss.png',
        log_scale=True
    )
    
    # Plot accuracy curves
    plot_learning_curves(
        [combined_dfs[combined_dfs['params'] == p] for p in combined_dfs['params'].unique()],
        'Combined Momentum and Learning Rate Decay',
        'Accuracy (%)',
        'part2d_combined_accuracy.png'
    )

def analyze_adam():
    """Analyze Adam results."""
    # Load results
    adam_dfs = load_results('results_adam_*.csv')
    
    # Plot learning curves
    plot_learning_curves(
        [adam_dfs[adam_dfs['params'] == p] for p in adam_dfs['params'].unique()],
        'Adam Learning Curves',
        'Loss',
        'part3_adam_loss.png',
        log_scale=True
    )
    
    # Plot accuracy curves
    plot_learning_curves(
        [adam_dfs[adam_dfs['params'] == p] for p in adam_dfs['params'].unique()],
        'Adam Accuracy Curves',
        'Accuracy (%)',
        'part3_adam_accuracy.png'
    )
    
    # Plot hyperparameter sweeps
    for param in ['beta1', 'beta2', 'epsilon', 'lr']:
        param_dfs = adam_dfs[adam_dfs['params'].str.contains(param)]
        plot_hyperparameter_sweep(
            param_dfs,
            param,
            f'Adam {param.upper()} Sweep',
            f'part3_adam_{param}_sweep.png'
        )

def analyze_rmsprop():
    """Analyze RMSProp results."""
    # Load results
    rmsprop_dfs = load_results('results_rmsprop_*.csv')
    
    # Plot learning curves
    plot_learning_curves(
        [rmsprop_dfs[rmsprop_dfs['params'] == p] for p in rmsprop_dfs['params'].unique()],
        'RMSProp Learning Curves',
        'Loss',
        'part3_rmsprop_loss.png',
        log_scale=True
    )
    
    # Plot accuracy curves
    plot_learning_curves(
        [rmsprop_dfs[rmsprop_dfs['params'] == p] for p in rmsprop_dfs['params'].unique()],
        'RMSProp Accuracy Curves',
        'Accuracy (%)',
        'part3_rmsprop_accuracy.png'
    )
    
    # Plot hyperparameter sweeps
    for param in ['rho', 'epsilon', 'lr']:
        param_dfs = rmsprop_dfs[rmsprop_dfs['params'].str.contains(param)]
        plot_hyperparameter_sweep(
            param_dfs,
            param,
            f'RMSProp {param.upper()} Sweep',
            f'part3_rmsprop_{param}_sweep.png'
        )

def generate_summary_table():
    """Generate a summary table of best results for each method."""
    methods = {
        'SGD': 'results_sgd_*.csv',
        'SGD+Momentum': 'results_sgd_momentum_*.csv',
        'SGD+LR Decay': 'results_sgd_lr_decay_*.csv',
        'SGD+Momentum+LR Decay': 'results_sgd_momentum_lr_decay_*.csv',
        'Adam': 'results_adam_*.csv',
        'RMSProp': 'results_rmsprop_*.csv'
    }
    
    summary = []
    for method, pattern in methods.items():
        dfs = load_results(pattern)
        best_params = dfs.groupby('params')['value'].max().idxmax()
        best_acc = dfs[dfs['params'] == best_params]['value'].max()
        summary.append({
            'Method': method,
            'Best Parameters': best_params,
            'Best Accuracy (%)': best_acc
        })
    
    # Create summary table
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('optimizer_summary.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Method'], summary_df['Best Accuracy (%)'])
    plt.xticks(rotation=45)
    plt.title('Comparison of Optimization Methods')
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run all analyses."""
    print("Analyzing results...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Run analyses
    analyze_sgd()
    analyze_momentum()
    analyze_lr_decay()
    analyze_combined()
    analyze_adam()
    analyze_rmsprop()
    
    # Generate summary
    generate_summary_table()
    
    print("Analysis complete. Results saved in 'plots' directory.")

if __name__ == '__main__':
    main() 