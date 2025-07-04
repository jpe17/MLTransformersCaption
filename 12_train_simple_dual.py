from model_explained import VisionLanguageEncoder, CaptionDecoder
from loader_dual_caption import get_dual_caption_data
import torch
import torch.nn as nn
import itertools
import os

def train_simple_dual_models():
    """
    Train two separate models:
    1. Positive model - trained only on positive (Bes) captions
    2. Negative model - trained only on negative (Anti-Bes) captions
    
    Much simpler than the complex dual model approach!
    """
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 1. Data Loading ---
    print("Loading dual caption data...")
    train_loader_fn, val_loader_fn = get_dual_caption_data(
        max_samples=6000,
        val_split=0.15,
        batch_size=4
    )
    
    # --- 2. Load Pre-trained Weights ---
    pretrained_path = "saved_models/explained/explained_20_epochs.pth"
    pretrained_checkpoint = None
    
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}")
        pretrained_checkpoint = torch.load(pretrained_path, map_location='cpu')
        print("✅ Pre-trained weights loaded")
    else:
        print(f"⚠️  Pre-trained weights not found at {pretrained_path}")
        print("Training from scratch...")
    
    # --- 3. Train Positive Model ---
    print("\n" + "="*50)
    print("🌟 TRAINING POSITIVE MODEL")
    print("="*50)
    
    pos_encoder = VisionLanguageEncoder().to(device)
    pos_decoder = CaptionDecoder().to(device)
    
    # Load pre-trained weights for positive model
    if pretrained_checkpoint:
        pos_encoder.load_state_dict(pretrained_checkpoint['encoder_state_dict'])
        pos_decoder.load_state_dict(pretrained_checkpoint['decoder_state_dict'])
        print("✅ Loaded pre-trained weights for positive model")
    
    # Train positive model
    train_single_model(pos_encoder, pos_decoder, train_loader_fn, val_loader_fn, 
                      caption_type="positive", device=device, num_epochs=10)
    
    # Save positive model
    os.makedirs("saved_models/simple_dual", exist_ok=True)
    torch.save({
        'encoder_state_dict': pos_encoder.state_dict(),
        'decoder_state_dict': pos_decoder.state_dict(),
    }, "saved_models/simple_dual/positive_model.pth")
    print("✅ Positive model saved")
    
    # --- 4. Train Negative Model ---
    print("\n" + "="*50)
    print("😠 TRAINING NEGATIVE MODEL")
    print("="*50)
    
    neg_encoder = VisionLanguageEncoder().to(device)
    neg_decoder = CaptionDecoder().to(device)
    
    # Load pre-trained weights for negative model
    if pretrained_checkpoint:
        neg_encoder.load_state_dict(pretrained_checkpoint['encoder_state_dict'])
        neg_decoder.load_state_dict(pretrained_checkpoint['decoder_state_dict'])
        print("✅ Loaded pre-trained weights for negative model")
    
    # Train negative model
    train_single_model(neg_encoder, neg_decoder, train_loader_fn, val_loader_fn, 
                      caption_type="negative", device=device, num_epochs=10)
    
    # Save negative model
    torch.save({
        'encoder_state_dict': neg_encoder.state_dict(),
        'decoder_state_dict': neg_decoder.state_dict(),
    }, "saved_models/simple_dual/negative_model.pth")
    print("✅ Negative model saved")
    
    print("\n🎉 Both models trained successfully!")
    print("Use inference_simple_dual.py to generate captions")

def train_single_model(encoder, decoder, train_loader_fn, val_loader_fn, caption_type, device, num_epochs=10):
    """Train a single model on either positive or negative captions"""
    
    # Setup trainable parameters (same as original)
    trainable_params = list(encoder.image_adapter.parameters()) + \
                       list(encoder.modality_embedding.parameters()) + \
                       list(decoder.vision_cross_attention.parameters()) + \
                       list(decoder.attn_layer_norm.parameters())
    
    # Add last 2 layers of Qwen
    if hasattr(decoder.qwen_model.model, 'layers'):
        for layer in decoder.qwen_model.model.layers[-2:]:
            trainable_params.extend([p for p in layer.parameters() if p.requires_grad])
    
    print(f"Training {sum(p.numel() for p in trainable_params):,} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        encoder.train()
        decoder.train()
        train_batches = train_loader_fn()
        
        epoch_loss = 0
        step_count = 0
        
        for step, (pil_images, pos_captions, neg_captions) in enumerate(train_batches):
            
            # Choose captions based on model type
            if caption_type == "positive":
                captions = pos_captions
            else:  # negative
                captions = neg_captions
            
            # Forward pass
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                image_embeddings, text_embeddings, target_tokens = encoder(pil_images, captions)
                logits, loss = decoder(image_embeddings, text_embeddings, target_tokens)
            
            if loss is None:
                print("Warning: Loss is None. Skipping step.")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            epoch_loss += loss.item()
            step_count += 1
            
            if step % 10 == 0:
                print(f"  Step {step}: Loss: {loss.item():.4f}")
        
        # Epoch summary
        if step_count > 0:
            avg_loss = epoch_loss / step_count
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    # Validation
    print(f"\n✅ {caption_type.title()} model training complete! Running validation...")
    encoder.eval()
    decoder.eval()
    
    val_batches = val_loader_fn()
    
    with torch.no_grad():
        for i, (pil_images, pos_captions, neg_captions) in enumerate(itertools.islice(val_batches, 2)):
            print(f"\n--- Validation Example {i+1} ---")
            
            # Choose true captions based on model type
            if caption_type == "positive":
                true_captions = pos_captions
            else:
                true_captions = neg_captions
            
            # Generate caption
            first_image = [pil_images[0]]
            image_embeddings, _, _ = encoder(first_image, ["a photo"])
            
            # Simple generation (using the same logic as original model)
            tokenizer = decoder.tokenizer
            start_tokens = tokenizer("a", return_tensors="pt", add_special_tokens=False)
            input_ids = start_tokens['input_ids'].to(device)
            
            generated_ids = []
            
            for _ in range(20):  # Max length
                text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
                
                # Add text modality embedding
                text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=device)
                text_mod_embs = encoder.modality_embedding(text_ids)
                text_embeddings = text_embeddings + text_mod_embs
                
                # Get logits
                logits, _ = decoder(image_embeddings, text_embeddings)
                next_token_logits = logits[:, -1, :] / 0.8
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, 40)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Decode
            if len(generated_ids) > 0:
                generated_caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                generated_caption = ""
            
            print(f"True {caption_type}: {true_captions[0]}")
            print(f"Generated: {generated_caption}")

if __name__ == "__main__":
    train_simple_dual_models() 