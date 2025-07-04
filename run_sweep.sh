#!/bin/bash

# Initialize and run wandb sweep for cross-attention model
echo "Starting wandb sweep for cross-attention image captioning model..."

# Make sure wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "wandb not found. Installing..."
    pip install wandb
fi

# Login to wandb (if not already logged in)
wandb login

# Initialize the sweep
echo "Initializing sweep..."
SWEEP_ID=$(wandb sweep sweep_config.yaml --project image-captioning-sweep 2>&1 | grep -o 'wandb agent.*' | cut -d' ' -f3)

echo "Sweep initialized with ID: $SWEEP_ID"
echo "View the sweep at: https://wandb.ai/sweep/$SWEEP_ID"

# Ask if user wants to run the agent
read -p "Do you want to start the sweep agent now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting sweep agent..."
    echo "Note: This will run multiple training runs. You can stop it with Ctrl+C"
    wandb agent $SWEEP_ID
else
    echo "To run the sweep later, use: wandb agent $SWEEP_ID"
    echo "Or run multiple agents in parallel with:"
    echo "  wandb agent $SWEEP_ID &"
    echo "  wandb agent $SWEEP_ID &"
fi 