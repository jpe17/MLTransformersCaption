from model_explained import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch
from transformers import get_linear_schedule_with_warmup
import itertools

def train_with_cross_attention():
    """
    Trains the image captioning model using a cross-attention mechanism
    for better fusion of image and text data.
    
    Key Changes from previous scripts:
    1.  Epoch-based Training: Runs for a fixed number of epochs (5) over the whole dataset.
    2.  Model Architecture: Uses the new VisionLanguageEncoder and CaptionDecoder
        from `model_explained.py` which implement cross-attention.
    3.  Cleaner Training Loop: Leverages the model's forward pass directly,
        making the code more modular and readable.
    4.  Refined Validation Loop: Adapted for the new model's architecture to
        correctly generate captions autoregressively.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Data and Model Setup ---
    print("Initializing model and data...")
    train_loader_fn, val_loader_fn = get_flickr_data()
    
    encoder = VisionLanguageEncoder().to(device)
    decoder = CaptionDecoder().to(device)
    
    # --- 2. Optimizer & Scheduler Setup ---
    # We will fine-tune the image adapter, modality embeddings, and the new cross-attention parts.
    trainable_params = list(encoder.image_adapter.parameters()) + \
                       list(encoder.modality_embedding.parameters()) + \
                       list(decoder.vision_cross_attention.parameters()) + \
                       list(decoder.attn_layer_norm.parameters()) + \
                       [p for p in decoder.qwen_model.model.layers[-2:].parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # --- 3. Training Loop (Epoch-based) ---
    num_epochs = 20
    print(f"Starting training for {num_epochs} epochs with Cross-Attention...")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Get a fresh data iterator for each epoch
        train_batches = train_loader_fn()
        
        encoder.train()
        decoder.train()
        
        for step, (pil_images, captions) in enumerate(train_batches):
            # Move data processing inside the model's forward pass for cleaner code
            image_embeddings, text_embeddings, target_tokens = encoder(pil_images, captions)
            
            # The target_tokens are already on the correct device from the encoder
            
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                logits, loss = decoder(image_embeddings, text_embeddings, target_tokens)
            
            if loss is None:
                print("Warning: Loss is None. Skipping step.")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            if step % 20 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")
            
    print("\n✅ Training complete! Running validation...")
    
    # --- 4. Validation Loop ---
    encoder.eval()
    decoder.eval()
    
    val_batches = val_loader_fn()

    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(itertools.islice(val_batches, 5)):
            first_image = [pil_images[0]]
            
            # --- Image Processing (via Encoder) ---
            # Pass a dummy caption to fulfill the encoder's signature
            image_embeddings, _, _ = encoder(first_image, ["dummy caption"])
            
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

                # Get logits using the cross-attention decoder
                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                    logits, _ = decoder(image_embeddings, text_embeddings_final)
                
                # Get the logit for the last token
                next_token_logits = logits[:, -1, :]
                
                # --- Top-k Sampling ---
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
    train_with_cross_attention() 