from model_no_modality import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch
from transformers import get_linear_schedule_with_warmup

def train_caption_model_no_modality():
    """
    Training script for the captioning model without modality embeddings.
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
                      [p for p in decoder.parameters() if p.requires_grad]
    
    # Use a much lower learning rate for fine-tuning to prevent destroying pretrained weights
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
    
    # Add a learning rate scheduler with warmup
    num_training_steps = 1000
    num_warmup_steps = 50
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    print("Starting image captioning training (no modality embedding)...")
    print(f"  - Learning Rate: {optimizer.defaults['lr']:.1e}")
    print(f"  - Warmup Steps: {num_warmup_steps}")
    
    # --- 3. Training Loop ---
    for step, (pil_images, captions) in enumerate(train_batches):
        if step >= num_training_steps:
            break
        
        encoder.train()
        decoder.train()
        
        batch_size = len(pil_images)
        with torch.no_grad():
            clip_inputs = encoder.clip_processor(images=pil_images, return_tensors="pt").to(device)
            vision_outputs = encoder.clip_model.vision_model(**clip_inputs)
            image_patches = vision_outputs.last_hidden_state[:, 1:, :]
        
        image_embeddings = encoder.image_adapter(image_patches)
        num_patches = image_embeddings.shape[1]
        
        image_input = image_embeddings
        
        tokenized = encoder.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True).to(device)
        input_ids = tokenized['input_ids']
        
        input_tokens = input_ids[:, :-1]
        target_tokens = input_ids[:, 1:]
        
        text_embeddings = encoder.qwen_model.get_input_embeddings()(input_tokens)
        text_input = text_embeddings
        
        combined_input = torch.cat([image_input, text_input], dim=1)
        
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            logits, loss = decoder(combined_input, target_tokens, num_patches)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
    print("\n✅ Training done! Testing on validation...")
    
    # --- 4. Validation with Manual Generation ---
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
            
            input_embeddings = image_embeddings
            
            # --- Manual Autoregressive Generation with Top-k Sampling---
            generated_ids = []
            for _ in range(25):
                attention_mask = torch.ones(input_embeddings.shape[:2], device=input_embeddings.device)
                
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    outputs = decoder.qwen_model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]

                k = 50
                top_k_logits, top_k_indices = torch.topk(logits, k)
                
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                
                if next_token_id.item() == decoder.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id.item())
                
                next_token_embedding = decoder.qwen_model.get_input_embeddings()(next_token_id)
                input_embeddings = torch.cat([input_embeddings, next_token_embedding], dim=1)

            predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nImage {i+1}:")
            print(f"  True: {true_captions[0]}")
            print(f"  Pred: {predicted_caption}")
            
    print("\nDone!")

if __name__ == "__main__":
    train_caption_model_no_modality() 