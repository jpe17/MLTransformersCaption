#!/usr/bin/env python3
"""
Script to initialize and run the wandb sweep for the cross-attention model.
"""

import wandb
import subprocess
import sys
import os

def main():
    # Make sure wandb is logged in
    try:
        wandb.login()
    except Exception as e:
        print(f"Error logging into wandb: {e}")
        print("Please run 'wandb login' first")
        sys.exit(1)
    
    # Initialize the sweep
    print("Initializing wandb sweep...")
    sweep_id = wandb.sweep(sweep="sweep_config.yaml", project="image-captioning-sweep")
    
    print(f"Sweep initialized with ID: {sweep_id}")
    print(f"You can view the sweep at: https://wandb.ai/sweep/{sweep_id}")
    
    # Ask user if they want to run the sweep agent
    response = input("Do you want to start the sweep agent now? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("Starting sweep agent...")
        print("Note: This will run multiple training runs. You can stop it with Ctrl+C")
        
        # Run the sweep agent
        try:
            wandb.agent(sweep_id, function=None, count=None)
        except KeyboardInterrupt:
            print("\nSweep agent stopped by user.")
    else:
        print(f"To run the sweep later, use: wandb agent {sweep_id}")
        print("Or run multiple agents in parallel with:")
        print(f"  wandb agent {sweep_id} &")
        print(f"  wandb agent {sweep_id} &")

if __name__ == "__main__":
    main() 