# Sweep Training and Model Management Guide

This guide explains how to use the hyperparameter sweep functionality and manage trained models.

## 1. Running a Sweep

### Start a New Sweep
```bash
python simple_sweep.py
```

This will:
- Create a new wandb sweep with predefined hyperparameter ranges
- Give you a sweep ID
- Ask if you want to start the agent immediately

### Run Sweep Agent
If you didn't start the agent immediately, you can run it later:
```bash
wandb agent YOUR_SWEEP_ID
```

## 2. Model Saving

Each sweep run automatically saves:
- **Model checkpoint**: `saved_models/sweep_runs/model_sweep_{run_id}_{optimizer}_lr{lr}_{scheduler}.pth`
- **Wandb artifact**: Uploaded to wandb for easy tracking
- **Complete config**: All hyperparameters and training metrics

### What's Saved in Each Checkpoint:
- `encoder_state_dict`: Vision-language encoder weights
- `decoder_state_dict`: Caption decoder weights
- `optimizer_state_dict`: Optimizer state
- `config`: All hyperparameters used
- `final_train_loss`: Final training loss
- `total_steps`: Total training steps
- `wandb_run_id`: Wandb run identifier

## 3. Using Trained Models for Inference

### Automatic Best Model Selection
```bash
# Uses the best model automatically (lowest loss)
python inference.py path/to/image.jpg
```

### Specify a Specific Model
```bash
# Use a specific model
python inference.py path/to/image.jpg saved_models/sweep_runs/model_sweep_abc123_adamw_lr2e-05_cosine.pth
```

### List Available Models
```bash
# See all available models and their performance
python inference.py --list-models
```

## 4. Analyzing Sweep Results

### Basic Analysis
```bash
# Analyze all sweep results
python analyze_sweep_results.py
```

This will show:
- Top 10 best performing models
- Summary statistics
- Optimizer and scheduler comparisons
- Visualization plots (if matplotlib is installed)

### Programmatic Access
```python
from analyze_sweep_results import analyze_sweep_results, get_best_model_path

# Get DataFrame with all results
df = analyze_sweep_results()

# Get path to best model
best_model = get_best_model_path()
```

## 5. Example Workflow

1. **Start a sweep**:
   ```bash
   python simple_sweep.py
   ```

2. **Let it run** (multiple models will be trained with different hyperparameters)

3. **Analyze results**:
   ```bash
   python analyze_sweep_results.py
   ```

4. **Test the best model**:
   ```bash
   python inference.py test_image.jpg
   ```

5. **Or test a specific model**:
   ```bash
   python inference.py --list-models
   python inference.py test_image.jpg saved_models/sweep_runs/model_sweep_xyz789_adam_lr3e-05_linear.pth
   ```

## 6. Sweep Configuration

The sweep tests these hyperparameters:
- **Optimizers**: AdamW, Adam
- **Learning rates**: 1e-5 to 5e-5 (log-uniform)
- **Weight decay**: 0.01 to 0.05 (uniform)
- **Schedulers**: Cosine, Linear, Constant
- **Gradient clipping**: 0.5 to 1.0
- **Generation parameters**: Top-k (20-100), Temperature (0.8-1.2)

## 7. Troubleshooting

### NaN Loss Issues
The training script includes automatic NaN detection and handling:
- Skips steps with NaN/Inf loss
- Checks gradient norms
- Uses conservative hyperparameter ranges

### Model Loading Issues
- Check if the model file exists
- Verify the checkpoint contains all required keys
- Use `--list-models` to see available models

### Wandb Issues
- Make sure you're logged in: `wandb login`
- Check your internet connection
- Verify project name in the sweep config

## 8. File Structure

```
saved_models/
└── sweep_runs/
    ├── model_sweep_abc123_adamw_lr2e-05_cosine.pth
    ├── model_sweep_def456_adam_lr1e-05_linear.pth
    └── ...
```

Each model file contains everything needed for inference and analysis. 