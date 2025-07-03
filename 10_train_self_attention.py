from model_self_attention import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch
from transformers import get_linear_schedule_with_warmup
import itertools

def train_self_attention_only():
    """
    Trains the image captioning model using SELF-ATTENTION ONLY but with all the 
    successful patterns from the explained model that made it work well.
    
    Key Improvements Applied:
    1. Epoch-based Training: Runs for a fixed number of epochs (5) over the whole dataset
    2. Enhanced Architecture: Uses improved positional embeddings and layer normalization
    3. Mixed Precision: Uses AMP for better performance and stability
    4. Proper Parameter Selection: Trains the right components for optimal results
    5. Clean Training Loop: Leverages the model's forward pass directly
    6. Improved Validation: Uses proper autoregressive generation with top-k sampling
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Data and Model Setup ---
    print("Initializing model and data...")
    train_loader_fn, val_loader_fn = get_flickr_data()
    
    encoder = VisionLanguageEncoder().to(device)
    decoder = CaptionDecoder().to(device)
    
    # --- 2. Optimizer & Scheduler Setup ---
    # Train the key components: image adapter, modality embeddings, enhanced positional embeddings, 
    # layer norm, and last 2 Qwen layers (same successful strategy as explained model)
    trainable_params = list(encoder.image_adapter.parameters()) + \
                       list(encoder.modality_embedding.parameters()) + \
                       list(decoder.enhanced_pos_embedding.parameters()) + \
                       list(decoder.input_layer_norm.parameters()) + \
                       [p for p in decoder.qwen_model.model.layers[-2:].parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
    
    # Use AMP for performance (same as explained model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # --- 3. Training Loop (Epoch-based like explained model) ---
    num_epochs = 5
    print(f"Starting training for {num_epochs} epochs with Self-Attention Only...")
    print("Using enhanced positional embeddings and layer normalization for better self-attention")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Get a fresh data iterator for each epoch
        train_batches = train_loader_fn()
        
        encoder.train()
        decoder.train()
        
        for step, (pil_images, captions) in enumerate(train_batches):
            # Clean data processing through encoder
            combined_embeddings, target_tokens, num_patches = encoder(pil_images, captions)
            
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                logits, loss, _ = decoder(combined_embeddings, target_tokens, num_patches)
            
            if loss is None:
                print("Warning: Loss is None. Skipping step.")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            if step % 20 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")
            
    print("\n✅ Training complete! Running validation...")
    
    # --- 4. Validation Loop (same clean approach as explained model) ---
    encoder.eval()
    decoder.eval()
    
    val_batches = val_loader_fn()

    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(itertools.islice(val_batches, 5)):
            first_image = [pil_images[0]]
            
            # --- Image Processing (via Encoder) ---
            # Pass a dummy caption to get image embeddings
            combined_embeddings, _, num_patches = encoder(first_image, ["dummy caption"])
            
            # Extract just the image part for generation
            image_embeddings = combined_embeddings[:, :num_patches, :]
            
            # --- Autoregressive Generation ---
            generated_ids = []
            # Start with the Beginning-Of-Sequence token ID
            sos_id = decoder.tokenizer.bos_token_id if decoder.tokenizer.bos_token_id is not None else decoder.tokenizer.eos_token_id
            input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)

            for _ in range(30): # Max caption length
                # Get text embeddings for the current sequence
                text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
                
                # Add text modality embedding
                text_mod_id = torch.ones_like(input_ids)
                text_mod_emb = encoder.modality_embedding(text_mod_id)
                text_embeddings_final = text_embeddings + text_mod_emb
                
                # Combine image and text embeddings
                combined_for_generation = torch.cat([image_embeddings, text_embeddings_final], dim=1)
                
                # Add enhanced positional embeddings and layer norm
                batch_size, seq_len, hidden_size = combined_for_generation.shape
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                pos_embeddings = decoder.enhanced_pos_embedding(position_ids)
                enhanced_embeddings = combined_for_generation + pos_embeddings
                enhanced_embeddings = decoder.input_layer_norm(enhanced_embeddings)

                # Get logits using the self-attention decoder
                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                    outputs = decoder.qwen_model(inputs_embeds=enhanced_embeddings)
                    logits = outputs.logits
                
                # Get the logit for the last token
                next_token_logits = logits[:, -1, :]
                
                # --- Top-k Sampling (same as explained model) ---
                k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                
                # Stop if EOS is generated
                if next_token_id.item() == decoder.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id.item())
                
                # Append the new token for the next iteration
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

            predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nImage {i+1}:")
            print(f"  True: {true_captions[0]}")
            print(f"  Pred: {predicted_caption}")
            
    print("\nDone!")

if __name__ == "__main__":
    train_self_attention_only() 