from model_cross_attention import ImageCaptionModel
from loader import get_flickr_data
from evaluate import print_eval
import torch

if __name__ == "__main__":
    # Load data
    train, val = get_flickr_data()
    train_batches = train()
    val_batches = val()
    
    # Initialize custom model (smaller for faster training)
    model = ImageCaptionModel(d_model=256, nhead=8, num_layers=8)
    
    # Optimizer with weight decay and learning rate schedule
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    print("Starting training with custom decoder...")
    
    # Train for 300 steps with better schedule
    for step, (pil_images, captions) in enumerate(train_batches):
        if step >= 10000:
            break
            
        # Forward pass
        logits, loss = model(pil_images, captions)
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\nTraining done! Testing on validation...")
    
    # Test on validation
    model.eval()
    
    all_predictions = []
    all_true_captions = []
    
    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(val_batches):
            if i >= 20:  # Test on 20 examples for better metrics
                break
                
            # Generate captions for first image only
            first_image = [pil_images[0]]
            predicted_captions = model(first_image)
            
            pred_caption = predicted_captions[0]
            true_caption = true_captions[0]
            
            all_predictions.append(pred_caption)
            all_true_captions.append(true_caption)
            
            if i < 10:  # Only print first 10 for readability
                print(f"\nImage {i+1}:")
                print(f"True: {true_caption}")
                print(f"Pred: {pred_caption}")
    
    # Print evaluation metrics
    print_eval(all_predictions, all_true_captions)
    
    print("\nDone!") 