from model_explained import VisionLanguageEncoder as VisionLanguageEncoderBase, CaptionDecoder
from loader import get_flickr_data
import torch
from transformers import get_linear_schedule_with_warmup
import itertools

class VisionLanguageEncoderNoModality(VisionLanguageEncoderBase):
    """
    Version without modality embeddings - removes the distinction between image and text tokens
    """
    def forward(self, pil_images, captions):
        # Get the device from the model
        device = next(self.qwen_model.parameters()).device
        
        # 1. Process images with CLIP to get patch embeddings
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt")
            # Move clip inputs to the correct device
            clip_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in clip_inputs.items()}
            vision_outputs = self.clip_model.vision_model(**clip_inputs)
            image_patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # Exclude CLS token

        # 2. Adapt CLIP embeddings to Qwen's dimension
        image_patch_embeddings = self.image_adapter(image_patch_embeddings)

        # 3. Process text captions to get input and target tokens
        tokenized = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        tokens = tokenized['input_ids'].to(device)
        
        sos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        eos = self.tokenizer.eos_token_id
        
        sos_tokens = torch.full((tokens.shape[0], 1), sos, dtype=tokens.dtype, device=device)
        input_padded = torch.cat([sos_tokens, tokens], dim=1)
        
        eos_tokens = torch.full((tokens.shape[0], 1), eos, dtype=tokens.dtype, device=device)
        target_padded = torch.cat([tokens, eos_tokens], dim=1)

        # 4. Get text embeddings from Qwen's embedding layer
        with torch.no_grad():
            text_embeddings = self.qwen_model.get_input_embeddings()(input_padded)

        # 5. NO MODALITY EMBEDDINGS - return embeddings as-is
        return image_patch_embeddings, text_embeddings, target_padded

def train_v2_01_no_modality():
    """
    V2_01: Training without modality embeddings
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("V2_01: Training WITHOUT modality embeddings")

    # --- 1. Data and Model Setup ---
    print("Initializing model and data...")
    train_loader_fn, val_loader_fn = get_flickr_data()
    
    encoder = VisionLanguageEncoderNoModality().to(device)
    decoder = CaptionDecoder().to(device)
    
    # --- 2. Optimizer & Scheduler Setup ---
    # We will fine-tune the image adapter and the new cross-attention parts (NO modality embeddings)
    trainable_params = list(encoder.image_adapter.parameters()) + \
                       list(decoder.vision_cross_attention.parameters()) + \
                       list(decoder.attn_layer_norm.parameters()) + \
                       [p for p in decoder.qwen_model.model.layers[-2:].parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # --- 3. Training Loop (Epoch-based) ---
    num_epochs = 20
    print(f"Starting training for {num_epochs} epochs...")
    
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
                
                # NO modality embeddings added here

                # Get logits using the cross-attention decoder
                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                    logits, _ = decoder(image_embeddings, text_embeddings)
                
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
            
    print("\nV2_01 Done!")

if __name__ == "__main__":
    train_v2_01_no_modality() 