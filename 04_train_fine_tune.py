from model import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch
from transformers import get_linear_schedule_with_warmup

def train_stable_caption_model():
    """
    A more stable training approach for fine-tuning, designed to prevent catastrophic forgetting.
    Key changes:
    1. Lower learning rate (2e-5) to make smaller, safer updates.
    2. Learning rate scheduler with a warmup period to gently adapt the model.
    3. Better generation parameters during validation to avoid repetitive loops.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Data and Model Setup ---
    print("Initializing model and data...")
    train, val = get_flickr_data()
    train_batches = train()
    val_batches = val()
    
    encoder = VisionLanguageEncoder().to(device)
    decoder = CaptionDecoder().to(device)
    
    # --- 2. Stable Optimizer & Scheduler Setup ---
    # Combine trainable parameters
    trainable_params = list(encoder.image_adapter.parameters()) + \
                      list(encoder.modality_embedding.parameters()) + \
                      [p for p in decoder.parameters() if p.requires_grad]
    
    # Use a much lower learning rate for fine-tuning to prevent destroying pretrained weights
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-6, weight_decay=0.01)
    
    # Add a learning rate scheduler with warmup
    num_training_steps = 1000  # Increased from 50 to 500 for meaningful learning
    num_warmup_steps = 50   # ~10% of total steps for warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    print("Starting STABLE image captioning training...")
    print(f"  - Learning Rate: {optimizer.defaults['lr']:.1e}")
    print(f"  - Warmup Steps: {num_warmup_steps}")
    
    # --- 3. Training Loop ---
    for step, (pil_images, captions) in enumerate(train_batches):
        if step >= num_training_steps:
            break
        
        encoder.train()
        decoder.train()
        
        # (The rest of the teacher-forcing logic is the same)
        batch_size = len(pil_images)
        with torch.no_grad():
            clip_inputs = encoder.clip_processor(images=pil_images, return_tensors="pt").to(device)
            vision_outputs = encoder.clip_model.vision_model(**clip_inputs)
            image_patches = vision_outputs.last_hidden_state[:, 1:, :]
        
        image_embeddings = encoder.image_adapter(image_patches)
        num_patches = image_embeddings.shape[1]
        
        image_ids = torch.zeros(batch_size, num_patches, dtype=torch.long, device=device)
        image_mod_embs = encoder.modality_embedding(image_ids)
        image_input = image_embeddings + image_mod_embs
        
        tokenized = encoder.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True).to(device)
        input_ids = tokenized['input_ids']
        
        input_tokens = input_ids[:, :-1]
        target_tokens = input_ids[:, 1:]
        
        text_embeddings = encoder.qwen_model.get_input_embeddings()(input_tokens)
        text_ids = torch.ones_like(input_tokens)
        text_mod_embs = encoder.modality_embedding(text_ids)
        text_input = text_embeddings + text_mod_embs
        
        combined_input = torch.cat([image_input, text_input], dim=1)
        
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            logits, loss = decoder(combined_input, target_tokens, num_patches)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Update the learning rate
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
    print("\n✅ Training done! Testing on validation...")
    
    # --- 4. Validation with CORRECTED Manual Generation ---
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(val_batches):
            if i >= 5:
                break
            
            # --- Image Processing ---
            first_image = [pil_images[0]]
            clip_inputs = encoder.clip_processor(images=first_image, return_tensors="pt").to(device)
            vision_outputs = encoder.clip_model.vision_model(**clip_inputs)
            image_patches = vision_outputs.last_hidden_state[:, 1:, :]
            image_embeddings = encoder.image_adapter(image_patches)
            
            batch_size, num_patches = 1, image_embeddings.shape[1]
            image_ids = torch.zeros(batch_size, num_patches, dtype=torch.long, device=image_embeddings.device)
            image_mod_embs = encoder.modality_embedding(image_ids)
            # This is our starting sequence of embeddings, containing only the image
            input_embeddings = image_embeddings + image_mod_embs
            
            # --- Manual Autoregressive Generation with Top-k Sampling---
            generated_ids = []
            for _ in range(25):  # Generate up to 25 tokens
                # Create an attention mask for the current input sequence length
                attention_mask = torch.ones(input_embeddings.shape[:2], device=input_embeddings.device)
                
                # Get logits from the decoder for the last token in the sequence
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    outputs = decoder.qwen_model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Shape: [1, vocab_size]

                # --- Top-k Sampling to prevent repetition ---
                k = 50
                top_k_logits, top_k_indices = torch.topk(logits, k)
                
                # Convert logits to probabilities and sample
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                
                # Stop if we generate the EOS token
                if next_token_id.item() == decoder.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id.item())
                
                # --- Prepare for the next loop iteration ---
                # Get the embedding for the newly generated token
                next_token_embedding = decoder.qwen_model.get_input_embeddings()(next_token_id)
                # **CRITICAL**: Add the text modality embedding, just like in training
                text_mod_embedding = encoder.modality_embedding(torch.ones_like(next_token_id).to(device))
                next_token_full_embedding = next_token_embedding + text_mod_embedding
                
                # Append the new token's full embedding to our input sequence
                input_embeddings = torch.cat([input_embeddings, next_token_full_embedding], dim=1)

            # Decode the final sequence of generated IDs
            predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nImage {i+1}:")
            print(f"  True: {true_captions[0]}")
            print(f"  Pred: {predicted_caption}")
            
    print("\nDone!")

if __name__ == "__main__":
    train_stable_caption_model() 