from model_explained import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch
from transformers import get_linear_schedule_with_warmup
import itertools
import wandb
import os

def train_with_cross_attention():
    """
    Trains the image captioning model using a cross-attention mechanism
    with wandb sweep support for hyperparameter optimization.
    """
    
    # Initialize wandb
    wandb.init()
    config = wandb.config
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Wandb config: {dict(config)}")
    
    # Check for problematic learning rates
    if config.learning_rate > 0.1:
        print(f"WARNING: Very high learning rate detected: {config.learning_rate}")
        print("This might cause training instability. Consider using a lower learning rate.")
        # Clamp to a safe maximum
        config.learning_rate = min(config.learning_rate, 0.001)
        print(f"Clamped learning rate to: {config.learning_rate}")
    
    if config.learning_rate < 1e-7:
        print(f"WARNING: Very low learning rate detected: {config.learning_rate}")
        print("This might cause very slow training.")
        # Clamp to a safe minimum
        config.learning_rate = max(config.learning_rate, 1e-6)
        print(f"Clamped learning rate to: {config.learning_rate}")

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

    # Use sweep parameters for optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            trainable_params, 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
    
    # Learning rate scheduler
    total_steps = config.num_epochs * 100  # Approximate steps per epoch
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif config.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
    else:  # constant
        scheduler = None
    
    # Use AMP for performance
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # --- 3. Training Loop (Epoch-based) ---
    num_epochs = config.num_epochs
    print(f"Starting training for {num_epochs} epochs with Cross-Attention...")
    
    total_loss = 0
    step_count = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Get a fresh data iterator for each epoch
        train_batches = train_loader_fn()
        
        encoder.train()
        decoder.train()
        
        epoch_loss = 0
        epoch_steps = 0
        
        for step, (pil_images, captions) in enumerate(train_batches):
            # Move data processing inside the model's forward pass for cleaner code
            image_embeddings, text_embeddings, target_tokens = encoder(pil_images, captions)
            
            # The target_tokens are already on the correct device from the encoder
            
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                logits, loss = decoder(image_embeddings, text_embeddings, target_tokens)
            
            if loss is None:
                print("Warning: Loss is None. Skipping step.")
                continue
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected: {loss.item()}. Skipping step.")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Check for NaN gradients before clipping
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(f"Warning: NaN or Inf gradient norm detected: {total_norm}. Skipping step.")
                # Reset the scaler state for next iteration
                scaler.update()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            epoch_loss += loss.item()
            step_count += 1
            epoch_steps += 1
            
            # Log to wandb
            if step % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "step": step_count
                })
            
            if step % 20 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch
        })
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            
    print("\n✅ Training complete! Running validation...")
    
    # --- 4. Validation Loop ---
    encoder.eval()
    decoder.eval()
    
    val_batches = val_loader_fn()
    
    validation_examples = []
    
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
                
                # --- Top-k Sampling with sweep parameters ---
                k = config.top_k
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                
                # Apply temperature
                if config.temperature > 0:
                    top_k_logits = top_k_logits / config.temperature
                
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
            
            validation_examples.append({
                "true_caption": true_captions[0],
                "predicted_caption": predicted_caption
            })
    
    # Log validation examples to wandb
    wandb.log({
        "validation_examples": wandb.Table(
            columns=["True Caption", "Predicted Caption"],
            data=[[ex["true_caption"], ex["predicted_caption"]] for ex in validation_examples]
        )
    })
    
    # Calculate and log final metrics
    avg_train_loss = total_loss / step_count if step_count > 0 else 0
    wandb.log({
        "final_train_loss": avg_train_loss,
        "total_steps": step_count
    })
    
    print(f"\nFinal average training loss: {avg_train_loss:.4f}")
    
    # --- 5. Save the trained model ---
    print("Saving model...")
    
    # Create save directory if it doesn't exist
    save_dir = "saved_models/sweep_runs"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create unique filename using wandb run id and key hyperparameters
    run_id = wandb.run.id
    optimizer_name = config.optimizer
    lr = config.learning_rate
    scheduler_name = config.scheduler
    
    model_filename = f"model_sweep_{run_id}_{optimizer_name}_lr{lr:.1e}_{scheduler_name}.pth"
    model_path = os.path.join(save_dir, model_filename)
    
    # Save model state dict, optimizer state, and config
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': dict(config),
        'final_train_loss': avg_train_loss,
        'total_steps': step_count,
        'wandb_run_id': run_id
    }
    
    torch.save(checkpoint, model_path)
    print(f"Model saved to: {model_path}")
    
    # Log the model path to wandb for easy tracking
    wandb.log({
        "model_path": model_path,
        "model_filename": model_filename
    })
    
    # Save model as wandb artifact for better organization
    model_artifact = wandb.Artifact(
        name=f"model_sweep_{run_id}",
        type="model",
        description=f"Trained model with {optimizer_name} optimizer, lr={lr:.1e}, scheduler={scheduler_name}"
    )
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    
    print("Done!")

if __name__ == "__main__":
    train_with_cross_attention() 