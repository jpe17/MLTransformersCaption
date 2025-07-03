from model import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch

# Simple training
if __name__ == "__main__":
    # Load data
    train, val = get_flickr_data()
    train_batches = train()
    val_batches = val()
    
    # Initialize models
    encoder = VisionLanguageEncoder()
    decoder = CaptionDecoder()
    
    # Only train encoder parameters (decoder is frozen)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)
    
    print("Starting training...")
    
    # Quick training - just 50 steps
    for step, (pil_images, captions) in enumerate(train_batches):
        if step >= 250:  # Quick training
            break
            
        # Forward pass
        final_input, target_tokens, num_patches = encoder(pil_images, captions)
        logits, loss = decoder(final_input, target_tokens, num_patches)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    print("\nTraining done! Testing on validation...")
    
    # Test on validation - show 10 predictions
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(val_batches):
            if i >= 10:  # Only show 10 examples
                break
                
            # Get image patches only (no text for prediction) - PROCESS ONLY FIRST IMAGE
            first_image = [pil_images[0]]  # Take only first image to avoid batch issues
            clip_inputs = encoder.clip_processor(images=first_image, return_tensors="pt")
            vision_outputs = encoder.clip_model.vision_model(**clip_inputs)
            image_patches = vision_outputs.last_hidden_state[:, 1:, :]  # Remove CLS
            image_embeddings = encoder.image_adapter(image_patches)
            
            # Add image modality embeddings
            batch_size, num_patches = image_embeddings.shape[0], image_embeddings.shape[1]  # Now batch_size = 1
            image_ids = torch.zeros(batch_size, num_patches, dtype=torch.long)
            image_mod_embs = encoder.modality_embedding(image_ids)
            image_input = image_embeddings + image_mod_embs
            
            # New approach: start with just SOS token, let cross-attention handle vision
            sos = encoder.tokenizer.bos_token_id or encoder.tokenizer.eos_token_id
            current_tokens = [sos]
            generated_tokens = []
            
            for _ in range(15):  # Generate up to 15 tokens
                # Create text embeddings for current sequence
                token_ids = torch.tensor([current_tokens])  # [1, seq_len]
                text_embeddings = encoder.qwen_model.get_input_embeddings()(token_ids)
                text_mod_emb = encoder.modality_embedding(torch.ones_like(token_ids))
                text_input = text_embeddings + text_mod_emb
                
                # Combine with image for decoder input
                combined = torch.cat([image_input, text_input], dim=1)
                
                # Generate next token
                logits, _ = decoder(combined, num_patches=num_patches)
                next_token = logits[:, -1:, :].argmax(dim=-1).item()  # Get last position
                
                if next_token == encoder.tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token)
                current_tokens.append(next_token)
            
            # Decode tokens to text
            predicted_caption = encoder.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"\nImage {i+1}:")
            print(f"True: {true_captions[0]}")
            print(f"Pred: {predicted_caption}")
            
    print("\nDone!") 