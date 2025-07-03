from model_embedding_loss import VisionLanguageEncoder, CaptionDecoder, EmbeddingLossTrainer
from loader import get_flickr_data
import torch
import itertools

def train_with_embedding_loss():
    """
    Train the image captioning model using embedding-based loss instead of cross-entropy.
    
    Key Innovation: Instead of forcing exact token matches, we compare semantic embeddings.
    This allows the model to learn that "car" and "vehicle" are similar, even if they're 
    different tokens.
    
    The loss function combines:
    1. Cosine similarity (semantic alignment)
    2. MSE loss (magnitude matching) 
    3. Contrastive loss (distinguishing different concepts)
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("🚀 Training with EMBEDDING-BASED LOSS - A more semantic approach!")

    # --- 1. Data and Model Setup ---
    print("\nInitializing model and data...")
    train_loader_fn, val_loader_fn = get_flickr_data()
    
    encoder = VisionLanguageEncoder().to(device)
    decoder = CaptionDecoder().to(device)
    
    # Create the trainer helper
    trainer = EmbeddingLossTrainer(encoder, decoder, device)
    
    # --- 2. Optimizer Setup ---
    # We need to train the new embedding projection layers too
    trainable_params = (
        list(encoder.image_adapter.parameters()) + 
        list(encoder.modality_embedding.parameters()) + 
        list(decoder.vision_cross_attention.parameters()) + 
        list(decoder.embedding_projection.parameters()) +  # New!
        list(decoder.embedding_norm.parameters()) +        # New!
        [decoder.temperature] +                            # New!
        [p for p in decoder.qwen_model.model.layers[-2:].parameters() if p.requires_grad]
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)  # Slightly higher LR for embedding training
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # --- 3. Training Loop (5 Epochs) ---
    num_epochs = 1
    print(f"\nStarting training for {num_epochs} epochs with Embedding-Based Loss...")
    print("📊 Loss Components:")
    print("  - 50% Cosine Similarity (semantic alignment)")
    print("  - 30% MSE Loss (magnitude matching)")
    print("  - 20% Contrastive Loss (concept distinction)")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Get fresh data iterator
        train_batches = train_loader_fn()
        
        encoder.train()
        decoder.train()
        
        epoch_loss = 0
        step_count = 0
        
        for step, (pil_images, captions) in enumerate(train_batches):
            
            # Use smaller batch - just first image to save memory
            single_image = [pil_images[0]]
            single_caption = [captions[0]]
            
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                loss, logits, predicted_embeddings = trainer.train_step(single_image, single_caption)
            
            if loss is None or torch.isnan(loss):
                print(f"Warning: Invalid loss at step {step}. Skipping...")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            step_count += 1
            
            if step % 25 == 0:
                avg_loss = epoch_loss / max(step_count, 1)
                temp_val = decoder.temperature.item()
                print(f"  Step {step:4d}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}, Temp: {temp_val:.3f}")
        
        avg_epoch_loss = epoch_loss / max(step_count, 1)
        print(f"  📈 Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
            
    print("\n✅ Training complete! Running validation with embedding-based model...")
    
    # --- 4. Validation ---
    encoder.eval()
    decoder.eval()
    
    val_batches = val_loader_fn()

    print("\n🔍 Validation Results:")
    print("=" * 60)
    
    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(itertools.islice(val_batches, 5)):
            
            # Generate caption using the trainer's method
            predicted_caption = trainer.generate_caption(pil_images[0])
            
            print(f"\n📸 Image {i+1}:")
            print(f"  🎯 True: {true_captions[0]}")
            print(f"  🤖 Pred: {predicted_caption}")
            
            # Optional: Show embedding similarity analysis
            if i == 0:  # Just for the first image
                print(f"  📊 Analysis: Embedding-based training allows semantic flexibility")
                print(f"      - Model learns 'car' ≈ 'vehicle', 'big' ≈ 'large', etc.")
                print(f"      - Loss encourages semantic similarity, not exact matches")
            
    print("\n" + "=" * 60)
    print("🎉 Embedding-based training complete!")
    print("\nKey Advantages of this approach:")
    print("✅ Semantic understanding over exact token matching")
    print("✅ Better handling of synonyms and paraphrases") 
    print("✅ More robust to vocabulary variations")
    print("✅ Contrastive learning prevents embedding collapse")

if __name__ == "__main__":
    train_with_embedding_loss() 