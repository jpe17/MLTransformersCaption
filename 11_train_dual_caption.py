from model_dual_caption import DualCaptionModel
from loader_dual_caption import get_dual_caption_data
import torch
import torch.nn as nn
import itertools
import os

def train_dual_caption_model():
    """
    Fine-tune the pre-trained model for dual caption generation (Bes + Anti-Bes).
    
    Key Features:
    1. Loads pre-trained weights from explained_20_epochs.pth
    2. Adds sentiment-aware components for positive/negative generation
    3. Efficient training with small batch sizes for 6000 images
    4. Dual loss calculation for both caption types
    """
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 1. Data Loading ---
    print("Loading dual caption data...")
    train_loader_fn, val_loader_fn = get_dual_caption_data(
        max_samples=6000,  # Your 6000 images
        val_split=0.15,    # Small validation set
        batch_size=4       # Small batch size for efficient training
    )
    
    # --- 2. Model Setup ---
    print("Initializing dual caption model...")
    model = DualCaptionModel().to(device)
    
    # Load pre-trained weights
    pretrained_path = "saved_models/explained/explained_20_epochs.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Load encoder weights (excluding sentiment_embedding)
            encoder_state = checkpoint['encoder_state_dict']
            encoder_dict = model.encoder.state_dict()
            
            # Filter out sentiment_embedding from pre-trained weights
            filtered_encoder_state = {k: v for k, v in encoder_state.items() 
                                    if k in encoder_dict and 'sentiment_embedding' not in k}
            encoder_dict.update(filtered_encoder_state)
            model.encoder.load_state_dict(encoder_dict)
            
            # Load decoder weights (excluding sentiment_projection)
            decoder_state = checkpoint['decoder_state_dict']
            decoder_dict = model.decoder.state_dict()
            
            # Filter out sentiment_projection from pre-trained weights
            filtered_decoder_state = {k: v for k, v in decoder_state.items() 
                                    if k in decoder_dict and 'sentiment_projection' not in k}
            decoder_dict.update(filtered_decoder_state)
            model.decoder.load_state_dict(decoder_dict)
            
            print("✅ Successfully loaded pre-trained weights (excluding new sentiment parameters)")
            
        except Exception as e:
            print(f"❌ Error loading pre-trained weights: {e}")
            print("Continuing with randomly initialized weights...")
    else:
        print(f"⚠️  Pre-trained weights not found at {pretrained_path}")
        print("Training from scratch...")
    
    # --- 3. Training Configuration ---
    # Collect trainable parameters
    new_sentiment_params = list(model.encoder.sentiment_embedding.parameters()) + \
                          list(model.decoder.sentiment_projection.parameters())
    
    existing_trainable_params = list(model.encoder.image_adapter.parameters()) + \
                               list(model.encoder.modality_embedding.parameters()) + \
                               list(model.decoder.vision_cross_attention.parameters()) + \
                               list(model.decoder.attn_layer_norm.parameters())
    
    # Add last 2 layers of Qwen
    if hasattr(model.decoder.qwen_model.model, 'layers'):
        for layer in model.decoder.qwen_model.model.layers[-2:]:
            existing_trainable_params.extend([p for p in layer.parameters() if p.requires_grad])
    
    all_trainable_params = new_sentiment_params + existing_trainable_params
    
    print(f"Training {sum(p.numel() for p in all_trainable_params):,} parameters")
    print(f"  - New sentiment parameters: {sum(p.numel() for p in new_sentiment_params):,}")
    print(f"  - Existing parameters: {sum(p.numel() for p in existing_trainable_params):,}")
    
    # Optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': new_sentiment_params, 'lr': 5e-4},        # Higher LR for new parameters
        {'params': existing_trainable_params, 'lr': 1e-5}    # Lower LR for pre-trained parameters
    ], weight_decay=0.01)
    
    # AMP for efficiency
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
    # --- 4. Training Loop ---
    num_epochs = 30  # Fewer epochs since we're fine-tuning
    print(f"Starting dual caption training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        model.train()
        train_batches = train_loader_fn()
        
        epoch_pos_loss = 0
        epoch_neg_loss = 0
        step_count = 0
        
        for step, (pil_images, pos_captions, neg_captions) in enumerate(train_batches):
            
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                # Forward pass for both positive and negative captions
                pos_logits, pos_loss, neg_logits, neg_loss = model(
                    pil_images, pos_captions, neg_captions
                )
                
                # Combined loss with equal weighting
                if pos_loss is not None and neg_loss is not None:
                    total_loss = pos_loss + neg_loss
                else:
                    print("Warning: One of the losses is None. Skipping step.")
                    continue
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            if pos_loss is not None and neg_loss is not None:
                epoch_pos_loss += pos_loss.item()
                epoch_neg_loss += neg_loss.item()
                step_count += 1
                
                if step % 10 == 0:
                    print(f"  Step {step}: Pos Loss: {pos_loss.item():.4f}, "
                          f"Neg Loss: {neg_loss.item():.4f}, "
                          f"Total: {total_loss.item():.4f}")
        
        # Epoch summary
        if step_count > 0:
            avg_pos_loss = epoch_pos_loss / step_count
            avg_neg_loss = epoch_neg_loss / step_count
            print(f"Epoch {epoch+1} - Avg Pos Loss: {avg_pos_loss:.4f}, "
                  f"Avg Neg Loss: {avg_neg_loss:.4f}")
    
    print("\n✅ Training complete! Running validation...")
    
    # --- 5. Validation ---
    model.eval()
    val_batches = val_loader_fn()
    
    with torch.no_grad():
        for i, (pil_images, pos_captions, neg_captions) in enumerate(itertools.islice(val_batches, 3)):
            print(f"\n--- Validation Example {i+1} ---")
            
            # Take first image from batch
            test_image = [pil_images[0]]
            
            # Generate both positive and negative captions
            pos_generated = model.generate_caption(test_image, sentiment_type="positive", max_length=25)
            neg_generated = model.generate_caption(test_image, sentiment_type="negative", max_length=25)
            
            print(f"True Positive: {pos_captions[0]}")
            print(f"Generated Positive: {pos_generated[0]}")
            print(f"True Negative: {neg_captions[0]}")
            print(f"Generated Negative: {neg_generated[0]}")
    
    # --- 6. Save Model ---
    print("\n💾 Saving trained model...")
    os.makedirs("saved_models/dual_caption", exist_ok=True)
    
    save_path = "saved_models/dual_caption/dual_caption_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': 4,
            'learning_rates': [5e-4, 1e-5],
            'data_samples': 6000
        }
    }, save_path)
    
    print(f"✅ Model saved to {save_path}")
    print("\nDone! You can now use the model for dual caption generation.")

if __name__ == "__main__":
    train_dual_caption_model() 