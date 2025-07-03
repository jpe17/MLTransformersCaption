# Ultra-Minimal Flickr30k DataLoader

Super simple Flickr30k data loader in just **75 lines**. No classes, downloads automatically, train/val splits.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from simple_flickr import get_dataloaders

# Get train and val generators
train_gen, val_gen, train_steps, val_steps = get_dataloaders(
    batch_size=16, 
    max_samples=1000,
    val_split=0.2
)

# Training loop
for images, captions in train_gen():
    # images: torch.Tensor [16, 3, 224, 224] - ready for CLIP!
    break

# Validation loop  
for images, captions in val_gen():
    # Validate here
    break
```

## Configuration

```python
train_gen, val_gen, train_steps, val_steps = get_dataloaders(
    batch_size=32,        # Batch size
    max_samples=None,     # Limit samples (None = all ~158k)
    val_split=0.2,        # 20% for validation
    num_workers=4         # Parallel workers (not used in generators)
)
```

## Example Output

```
Train steps: 25, Val steps: 6
✅ torch.Size([16, 3, 224, 224]) | Caption: A man sitting on a bench...
```

Run `python test.py` to see it in action! 