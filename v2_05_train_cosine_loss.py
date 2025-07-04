from model_explained import VisionLanguageEncoder, CaptionDecoder as CaptionDecoderBase
from loader import get_flickr_data
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import itertools
import wandb
import os
from PIL import Image

class CaptionDecoderCosineLoss(CaptionDecoderBase):
    """
    Version that uses cosine similarity loss instead of cross entropy
    """
    def forward(self, image_embeddings, text_embeddings, target_tokens=None):
        # image_embeddings: [batch_size, num_patches, hidden_size]
        # text_embeddings: [batch_size, seq_len, hidden_size]
        
        # 1. CROSS-ATTENTION: Text embeddings (query) attend to image embeddings (key/value)
        attn_output, _ = self.vision_cross_attention(
            query=text_embeddings,
            key=image_embeddings,
            value=image_embeddings
        )

        # 2. FUSION: Combine text and attention output with a residual connection and normalization
        fused_embeddings = self.attn_layer_norm(text_embeddings + attn_output)

        # 3. DECODING: Pass the fused embeddings through the causal language model
        outputs = self.qwen_model(inputs_embeds=fused_embeddings)
        logits = outputs.logits

        # 4. COSINE SIMILARITY LOSS (if training)
        loss = None
        if target_tokens is not None:
            # Get the target embeddings from the embedding layer
            target_embeddings = self.qwen_model.get_input_embeddings()(target_tokens)
            
            # Shift embeddings for causal language modeling
            # fused_embeddings: [batch_size, seq_len, hidden_size]
            # target_embeddings: [batch_size, seq_len, hidden_size]
            shift_predicted = fused_embeddings[..., :-1, :].contiguous()
            shift_target = target_embeddings[..., 1:, :].contiguous()
            
            # Flatten for cosine similarity calculation
            shift_predicted_flat = shift_predicted.view(-1, shift_predicted.size(-1))
            shift_target_flat = shift_target.view(-1, shift_target.size(-1))
            
            # Create mask to ignore padding tokens
            shift_labels = target_tokens[..., 1:].contiguous()
            mask = (shift_labels != self.tokenizer.pad_token_id).view(-1)
            
            if mask.sum() > 0:
                # Apply mask
                shift_predicted_masked = shift_predicted_flat[mask]
                shift_target_masked = shift_target_flat[mask]
                
                # Calculate cosine similarity
                cosine_sim = F.cosine_similarity(shift_predicted_masked, shift_target_masked, dim=-1)
                
                # Convert to loss (1 - cosine_similarity for minimization)
                loss = 1 - cosine_sim.mean()
            else:
                loss = torch.tensor(0.0, device=target_tokens.device, requires_grad=True)
            
        return logits, loss

def train_v2_05_cosine_loss():
    """
    V2_05: Training with cosine similarity loss instead of cross entropy
    """
    wandb.init(project="image-captioning-v2-experiments", name="v2_05_cosine_loss", config={
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "num_epochs": 3,
        "grad_clip_norm": 1.0,
        "model": "v2_05_cosine_loss"
    })
    config = wandb.config
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("V2_05: Training with COSINE SIMILARITY LOSS")

    # --- 1. Data and Model Setup ---
    print("Initializing model and data...")
    train_loader_fn, val_loader_fn = get_flickr_data()
    
    encoder = VisionLanguageEncoder().to(device)
    decoder = CaptionDecoderCosineLoss().to(device)
    
    # --- 2. Optimizer & Scheduler Setup ---
    # We will fine-tune the image adapter, modality embeddings, and the new cross-attention parts.
    trainable_params = list(encoder.image_adapter.parameters()) + \
                       list(encoder.modality_embedding.parameters()) + \
                       list(decoder.vision_cross_attention.parameters()) + \
                       list(decoder.attn_layer_norm.parameters()) + \
                       [p for p in decoder.qwen_model.model.layers[-2:].parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # --- 3. Training Loop (Epoch-based) ---
    num_epochs = config.num_epochs
    print(f"Starting training for {num_epochs} epochs with Cosine Loss...")
    
    total_loss = 0
    step_count = 0
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
            torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            step_count += 1
            if step % 20 == 0:
                print(f"  Step {step}, Cosine Loss: {loss.item():.4f}")
                wandb.log({
                    "train_loss_step": loss.item(),
                    "epoch": epoch
                })
            
    print("\n✅ Training complete! Running validation...")
    
    # --- 4. Validation Loop ---
    encoder.eval()
    decoder.eval()

    # --- Quantitative Validation (Loss) ---
    print("Calculating validation loss...")
    val_loss = 0
    val_steps = 0
    val_batches_for_loss = val_loader_fn()
    with torch.no_grad():
        for step, (pil_images, captions) in enumerate(itertools.islice(val_batches_for_loss, 20)):
            image_embeddings, text_embeddings, target_tokens = encoder(pil_images, captions)
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                _, v_loss = decoder(image_embeddings, text_embeddings, target_tokens)
            if v_loss is not None and not torch.isnan(v_loss) and not torch.isinf(v_loss):
                val_loss += v_loss.item()
                val_steps += 1
    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    print(f"  Average validation loss over {val_steps} batches: {avg_val_loss:.4f}")
    
    val_batches = val_loader_fn()
    validation_table = wandb.Table(columns=["Image", "True Caption", "Predicted Caption"])

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
            validation_table.add_data(wandb.Image(first_image[0]), true_captions[0], predicted_caption)

    avg_train_loss = total_loss / step_count if step_count > 0 else 0
    wandb.log({
        "final_avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "validation_examples": validation_table
    })
            
    print(f"\nFinal Average Training Loss: {avg_train_loss:.4f}")

    # --- Inference on a specific image ---
    print("\n--- Running inference on test_01.png ---")
    try:
        image_path = "test_01.png"
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping inference on this image.")
        else:
            pil_image = Image.open(image_path).convert("RGB")
            
            image_embeddings, _, _ = encoder([pil_image], ["dummy caption"])
            generated_ids = []
            sos_id = decoder.tokenizer.bos_token_id or decoder.tokenizer.eos_token_id
            input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)
            for _ in range(30):
                text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
                text_mod_id = torch.ones_like(input_ids)
                text_mod_emb = encoder.modality_embedding(text_mod_id)
                text_embeddings_final = text_embeddings + text_mod_emb
                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                    logits, _ = decoder(image_embeddings, text_embeddings_final)
                next_token_logits = logits[:, -1, :]
                k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
                if next_token_id.item() == decoder.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nImage: {image_path}")
            print(f"  Predicted Caption: {predicted_caption}")
            
            wandb.log({
                "test_inference_image": wandb.Image(pil_image, caption=f"Predicted: {predicted_caption}")
            })

    except Exception as e:
        print(f"An error occurred during inference on {image_path}: {e}")
            
    print("\nV2_05 Done!")

if __name__ == "__main__":
    train_v2_05_cosine_loss() 