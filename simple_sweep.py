#!/usr/bin/env python3
import wandb

# Simple sweep configuration as a Python dictionary
sweep_config = {
    'program': '08_train_explained_wandb.py',
    'method': 'bayes',
    'metric': {
        'goal': 'minimize',
        'name': 'final_train_loss'
    },
    'parameters': {
        'num_epochs': {'value': 5},
        'optimizer': {'values': ['adamw', 'adam']},
        'learning_rate': {'min': 1e-6, 'max': 1e-4, 'distribution': 'log_uniform'},
        'weight_decay': {'min': 0.001, 'max': 0.1, 'distribution': 'log_uniform'},
        'beta1': {'min': 0.85, 'max': 0.95, 'distribution': 'uniform'},
        'beta2': {'min': 0.99, 'max': 0.999, 'distribution': 'uniform'},
        'momentum': {'min': 0.85, 'max': 0.95, 'distribution': 'uniform'},
        'scheduler': {'values': ['cosine', 'linear', 'constant']},
        'grad_clip_norm': {'min': 0.5, 'max': 2.0, 'distribution': 'uniform'},
        'top_k': {'min': 20, 'max': 100, 'distribution': 'int_uniform'},
        'temperature': {'min': 0.7, 'max': 1.3, 'distribution': 'uniform'}
    }
}

def main():
    print("Initializing sweep...")
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='image-captioning-sweep')
    
    print(f"Sweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/sweep/{sweep_id}")
    
    # Ask if user wants to run the agent
    response = input("Start sweep agent now? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("Starting sweep agent...")
        print("Press Ctrl+C to stop")
        
        try:
            # Run the sweep agent with the training script
            import subprocess
            subprocess.run(['python', '-c', f'import wandb; wandb.agent("{sweep_id}")'])
        except KeyboardInterrupt:
            print("\nStopped by user")
    else:
        print(f"Run later with: wandb agent {sweep_id}")

if __name__ == "__main__":
    main() 