from model_self_attention import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch
from transformers import get_linear_schedule_with_warmup
import itertools
import wandb
from pathlib import Path

def train_self_attention_only():
    """
    Same as the original 10_train_self_attention.py but with wandb logging and model saving.
    """
    
    # Initialize wandb
    wandb.init(project="image-captioning", name="self-attention-model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Data and Model Setup ---
    print("Initializing model and data...")
    train_loader_fn, val_loader_fn = get_flickr_data()
    
    encoder = VisionLanguageEncoder().to(device)
    decoder = CaptionDecoder().to(device)
    
    # --- 2. Optimizer & Scheduler Setup ---
    trainable_params = list(encoder.image_adapter.parameters()) + \
                       list(encoder.modality_embedding.parameters()) + \
                       list(decoder.enhanced_pos_embedding.parameters()) + \
                       list(decoder.input_layer_norm.parameters()) + \
                       [p for p in decoder.qwen_model.model.layers[-2:].parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # --- 3. Training Loop (Epoch-based) ---
    num_epochs = 1
    print(f"Starting training for {num_epochs} epochs with Self-Attention Only...")
    print("Using enhanced positional embeddings and layer normalization for better self-attention")
    
    global_step = 0
    
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
            
            # Log to wandb
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch + 1,
                "step": global_step
            })
            
            global_step += 1
            
            if step % 20 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")
            
    print("\n✅ Training complete! Saving model...")
    
    # Save model
    save_dir = Path("saved_models/self_attention")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_dir / "model_checkpoint.pth")
    
    print(f"Model saved to {save_dir / 'model_checkpoint.pth'}")
    
    # --- 4. Validation Loop ---
    print("Running validation...")
    encoder.eval()
    decoder.eval()
    
    val_batches = val_loader_fn()

    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(itertools.islice(val_batches, 5)):
            first_image = [pil_images[0]]
            
            # --- Image Processing (via Encoder) ---
            combined_embeddings, _, num_patches = encoder(first_image, ["dummy caption"])
            
            # Extract just the image part for generation
            image_embeddings = combined_embeddings[:, :num_patches, :]
            
            # --- Autoregressive Generation ---
            generated_ids = []
            sos_id = decoder.tokenizer.bos_token_id if decoder.tokenizer.bos_token_id is not None else decoder.tokenizer.eos_token_id
            input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)

            for _ in range(30): # Max caption length
                text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
                
                text_mod_id = torch.ones_like(input_ids)
                text_mod_emb = encoder.modality_embedding(text_mod_id)
                text_embeddings_final = text_embeddings + text_mod_emb
                
                combined_for_generation = torch.cat([image_embeddings, text_embeddings_final], dim=1)
                
                batch_size, seq_len, hidden_size = combined_for_generation.shape
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                pos_embeddings = decoder.enhanced_pos_embedding(position_ids)
                enhanced_embeddings = combined_for_generation + pos_embeddings
                enhanced_embeddings = decoder.input_layer_norm(enhanced_embeddings)

                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                    outputs = decoder.qwen_model(inputs_embeds=enhanced_embeddings)
                    logits = outputs.logits
                
                next_token_logits = logits[:, -1, :]
                
                k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                
                if next_token_id.item() == decoder.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

            predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nImage {i+1}:")
            print(f"  True: {true_captions[0]}")
            print(f"  Pred: {predicted_caption}")
            
            # Log sample to wandb
            wandb.log({
                f"sample_{i+1}_true": true_captions[0],
                f"sample_{i+1}_pred": predicted_caption
            })
            
    print("\nDone!")
    wandb.finish()

if __name__ == "__main__":
    train_self_attention_only() 