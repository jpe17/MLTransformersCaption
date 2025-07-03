from model import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import torch

def train_caption_model():
    """Proper training for image captioning with teacher forcing"""
    
    # Load data
    train, val = get_flickr_data()
    train_batches = train()
    val_batches = val()
    
    # Initialize models
    encoder = VisionLanguageEncoder()
    decoder = CaptionDecoder()
    
    # Train both encoder adapter and decoder unfrozen layers
    trainable_params = list(encoder.image_adapter.parameters()) + \
                      list(encoder.modality_embedding.parameters()) + \
                      [p for p in decoder.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    print("Starting PROPER image captioning training...")
    print("Using teacher forcing approach")
    
    for step, (pil_images, captions) in enumerate(train_batches):
        if step >= 20:  # Quick training
            break
        
        # Process images only (no captions in input!)
        batch_size = len(pil_images)
        
        # Get image embeddings
        with torch.no_grad():
            clip_inputs = encoder.clip_processor(images=pil_images, return_tensors="pt")
            vision_outputs = encoder.clip_model.vision_model(**clip_inputs)
            image_patches = vision_outputs.last_hidden_state[:, 1:, :]  # Remove CLS
        
        image_embeddings = encoder.image_adapter(image_patches)
        num_patches = image_embeddings.shape[1]
        
        # Add image modality embeddings
        image_ids = torch.zeros(batch_size, num_patches, dtype=torch.long)
        image_mod_embs = encoder.modality_embedding(image_ids)
        image_input = image_embeddings + image_mod_embs
        
        # Prepare captions for teacher forcing
        tokenized = encoder.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
        input_ids = tokenized['input_ids']  # [batch_size, seq_len]
        
        # Create input (without last token) and target (without first token)
        input_tokens = input_ids[:, :-1]  # [batch_size, seq_len-1] 
        target_tokens = input_ids[:, 1:]  # [batch_size, seq_len-1]
        
        # Get text embeddings for input tokens
        text_embeddings = encoder.qwen_model.get_input_embeddings()(input_tokens)
        text_ids = torch.ones_like(input_tokens)
        text_mod_embs = encoder.modality_embedding(text_ids)
        text_input = text_embeddings + text_mod_embs
        
        # Combine image + text for decoder
        combined_input = torch.cat([image_input, text_input], dim=1)
        
        # Forward pass through decoder
        logits, loss = decoder(combined_input, target_tokens, num_patches)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    print("\nTraining done! Testing on validation...")
    
    # Test on validation - generate captions from images only
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(val_batches):
            if i >= 5:  # Show fewer examples
                break
                
            # Process first image only
            first_image = [pil_images[0]]
            
            # Get image embeddings
            clip_inputs = encoder.clip_processor(images=first_image, return_tensors="pt")
            vision_outputs = encoder.clip_model.vision_model(**clip_inputs)
            image_patches = vision_outputs.last_hidden_state[:, 1:, :]
            image_embeddings = encoder.image_adapter(image_patches)
            
            batch_size, num_patches = 1, image_embeddings.shape[1]
            image_ids = torch.zeros(batch_size, num_patches, dtype=torch.long)
            image_mod_embs = encoder.modality_embedding(image_ids)
            image_input = image_embeddings + image_mod_embs
            
            # Generate caption token by token
            sos = encoder.tokenizer.bos_token_id or encoder.tokenizer.eos_token_id
            current_tokens = [sos]
            generated_tokens = []
            
            for _ in range(20):  # Generate up to 20 tokens
                # Create text input for current sequence
                token_ids = torch.tensor([current_tokens])
                text_embeddings = encoder.qwen_model.get_input_embeddings()(token_ids)
                text_mod_emb = encoder.modality_embedding(torch.ones_like(token_ids))
                text_input = text_embeddings + text_mod_emb
                
                # Combine with image
                combined = torch.cat([image_input, text_input], dim=1)
                
                # Get next token prediction
                logits, _ = decoder(combined, num_patches=num_patches)
                next_token = logits[:, -1:, :].argmax(dim=-1).item()
                
                if next_token == encoder.tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token)
                current_tokens.append(next_token)
            
            # Decode prediction
            predicted_caption = encoder.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"\nImage {i+1}:")
            print(f"True: {true_captions[0]}")
            print(f"Pred: {predicted_caption}")
    
    print("\nDone!")

if __name__ == "__main__":
    train_caption_model() 