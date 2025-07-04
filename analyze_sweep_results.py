#!/usr/bin/env python3
"""
Script to analyze and compare sweep results.
Usage: python analyze_sweep_results.py
"""

import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sweep_results():
    """Analyze all models from sweep runs and create comparison charts."""
    
    sweep_dir = Path("saved_models/sweep_runs")
    if not sweep_dir.exists():
        print("No sweep models directory found.")
        return
    
    results = []
    
    # Collect data from all models
    for model_file in sweep_dir.glob("*.pth"):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            config = checkpoint.get('config', {})
            
            result = {
                'filename': model_file.name,
                'run_id': checkpoint.get('wandb_run_id', 'N/A'),
                'final_train_loss': checkpoint.get('final_train_loss', float('inf')),
                'total_steps': checkpoint.get('total_steps', 0),
                'optimizer': config.get('optimizer', 'N/A'),
                'learning_rate': config.get('learning_rate', 0),
                'weight_decay': config.get('weight_decay', 0),
                'scheduler': config.get('scheduler', 'N/A'),
                'grad_clip_norm': config.get('grad_clip_norm', 0),
                'beta1': config.get('beta1', 0),
                'beta2': config.get('beta2', 0),
                'temperature': config.get('temperature', 0),
                'top_k': config.get('top_k', 0),
            }
            results.append(result)
            
        except Exception as e:
            print(f"Warning: Could not load {model_file}: {e}")
    
    if not results:
        print("No valid model files found.")
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Sort by loss (best first)
    df = df.sort_values('final_train_loss')
    
    print("="*80)
    print("SWEEP RESULTS ANALYSIS")
    print("="*80)
    
    # Show top 10 models
    print("\nTop 10 Best Models:")
    print("-"*80)
    top_10 = df.head(10)
    for idx, row in top_10.iterrows():
        print(f"Rank {idx+1}: {row['filename']}")
        print(f"  Loss: {row['final_train_loss']:.4f}")
        print(f"  Optimizer: {row['optimizer']}, LR: {row['learning_rate']:.1e}")
        print(f"  Scheduler: {row['scheduler']}, Grad Clip: {row['grad_clip_norm']:.2f}")
        print(f"  Run ID: {row['run_id']}")
        print("-"*40)
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total models: {len(df)}")
    print(f"Best loss: {df['final_train_loss'].min():.4f}")
    print(f"Worst loss: {df['final_train_loss'].max():.4f}")
    print(f"Average loss: {df['final_train_loss'].mean():.4f}")
    print(f"Median loss: {df['final_train_loss'].median():.4f}")
    
    # Optimizer comparison
    print(f"\nOptimizer Performance:")
    opt_stats = df.groupby('optimizer')['final_train_loss'].agg(['count', 'mean', 'min', 'max'])
    print(opt_stats)
    
    # Scheduler comparison
    print(f"\nScheduler Performance:")
    sched_stats = df.groupby('scheduler')['final_train_loss'].agg(['count', 'mean', 'min', 'max'])
    print(sched_stats)
    
    # Create visualizations if matplotlib is available
    try:
        create_visualizations(df)
    except ImportError:
        print("\nNote: Install matplotlib and seaborn for visualizations")
    
    return df

def create_visualizations(df):
    """Create visualization plots for the sweep results."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sweep Results Analysis', fontsize=16)
    
    # 1. Loss distribution
    axes[0, 0].hist(df['final_train_loss'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Final Training Loss')
    axes[0, 0].set_xlabel('Final Training Loss')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Optimizer comparison
    if len(df['optimizer'].unique()) > 1:
        df.boxplot(column='final_train_loss', by='optimizer', ax=axes[0, 1])
        axes[0, 1].set_title('Loss by Optimizer')
        axes[0, 1].set_xlabel('Optimizer')
        axes[0, 1].set_ylabel('Final Training Loss')
    
    # 3. Learning rate vs loss
    axes[1, 0].scatter(df['learning_rate'], df['final_train_loss'], alpha=0.6)
    axes[1, 0].set_title('Learning Rate vs Loss')
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Final Training Loss')
    axes[1, 0].set_xscale('log')
    
    # 4. Scheduler comparison
    if len(df['scheduler'].unique()) > 1:
        df.boxplot(column='final_train_loss', by='scheduler', ax=axes[1, 1])
        axes[1, 1].set_title('Loss by Scheduler')
        axes[1, 1].set_xlabel('Scheduler')
        axes[1, 1].set_ylabel('Final Training Loss')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = "sweep_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    # Show the plot
    plt.show()

def get_best_model_path():
    """Get the path to the best performing model."""
    df = analyze_sweep_results()
    if df is not None and len(df) > 0:
        best_model = df.iloc[0]
        model_path = f"saved_models/sweep_runs/{best_model['filename']}"
        print(f"\nBest model path: {model_path}")
        return model_path
    return None

if __name__ == "__main__":
    analyze_sweep_results() 