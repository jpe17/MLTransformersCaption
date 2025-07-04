from model_explained import VisionLanguageEncoder, CaptionDecoder
import torch
import torch.nn as nn

class DualCaptionVisionLanguageEncoder(VisionLanguageEncoder):
    """
    Enhanced encoder that can handle both positive and negative caption generation.
    Adds a sentiment/style embedding to distinguish between Bes and Anti-Bes captions.
    """
    def __init__(self):
        super().__init__()
        
        # Add sentiment embedding: 0=Bes (positive), 1=Anti-Bes (negative)
        qwen_emb_dimension = self.qwen_model.config.hidden_size
        self.sentiment_embedding = nn.Embedding(2, qwen_emb_dimension)
        
        # Initialize sentiment embeddings to be different
        nn.init.normal_(self.sentiment_embedding.weight, mean=0, std=0.02)
    
    def forward(self, pil_images, captions, sentiment_ids=None):
        """
        Args:
            pil_images: List of PIL images
            captions: List of captions
            sentiment_ids: Tensor of shape [batch_size] with 0=Bes, 1=Anti-Bes
        """
        # Get the base image and text embeddings
        image_embeddings, text_embeddings, target_tokens = super().forward(pil_images, captions)
        
        # Add sentiment embeddings if provided
        if sentiment_ids is not None:
            sentiment_ids = sentiment_ids.to(image_embeddings.device)
            sentiment_embs = self.sentiment_embedding(sentiment_ids)  # [batch_size, hidden_size]
            
            # Add sentiment to text embeddings (broadcast across sequence length)
            text_embeddings = text_embeddings + sentiment_embs.unsqueeze(1)
        
        return image_embeddings, text_embeddings, target_tokens

class DualCaptionDecoder(CaptionDecoder):
    """
    Enhanced decoder that can generate both positive and negative captions.
    Uses the same cross-attention mechanism but with sentiment-aware embeddings.
    """
    def __init__(self):
        super().__init__()
        
        # Add a sentiment-aware projection layer
        hidden_size = self.qwen_model.config.hidden_size
        self.sentiment_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, image_embeddings, text_embeddings, target_tokens=None, sentiment_ids=None):
        """
        Args:
            image_embeddings: [batch_size, num_patches, hidden_size]
            text_embeddings: [batch_size, seq_len, hidden_size] (already has sentiment info)
            target_tokens: [batch_size, seq_len] for training
            sentiment_ids: [batch_size] with 0=Bes, 1=Anti-Bes (for additional processing)
        """
        
        # Apply sentiment-aware projection to text embeddings
        if sentiment_ids is not None:
            text_embeddings = self.sentiment_projection(text_embeddings)
        
        # Use the parent's forward method for cross-attention and decoding
        return super().forward(image_embeddings, text_embeddings, target_tokens)

class DualCaptionModel(nn.Module):
    """
    Complete dual caption model that combines encoder and decoder.
    Can generate both positive and negative captions for the same image.
    """
    def __init__(self):
        super().__init__()
        self.encoder = DualCaptionVisionLanguageEncoder()
        self.decoder = DualCaptionDecoder()
        
    def forward(self, pil_images, pos_captions, neg_captions):
        """
        Training forward pass with both positive and negative captions.
        
        Args:
            pil_images: List of PIL images
            pos_captions: List of positive captions
            neg_captions: List of negative captions
            
        Returns:
            pos_logits, pos_loss, neg_logits, neg_loss
        """
        batch_size = len(pil_images)
        device = next(self.parameters()).device
        
        # Process positive captions
        pos_sentiment_ids = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0 = Bes
        pos_image_embs, pos_text_embs, pos_targets = self.encoder(pil_images, pos_captions, pos_sentiment_ids)
        pos_logits, pos_loss = self.decoder(pos_image_embs, pos_text_embs, pos_targets, pos_sentiment_ids)
        
        # Process negative captions
        neg_sentiment_ids = torch.ones(batch_size, dtype=torch.long, device=device)   # 1 = Anti-Bes
        neg_image_embs, neg_text_embs, neg_targets = self.encoder(pil_images, neg_captions, neg_sentiment_ids)
        neg_logits, neg_loss = self.decoder(neg_image_embs, neg_text_embs, neg_targets, neg_sentiment_ids)
        
        return pos_logits, pos_loss, neg_logits, neg_loss
    
    def generate_caption(self, pil_images, sentiment_type="positive", max_length=30, temperature=0.7, top_k=50):
        """
        Generate captions for inference - simplified approach.
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            batch_size = len(pil_images)
            
            # Set sentiment
            sentiment_id = 0 if sentiment_type == "positive" else 1
            sentiment_ids = torch.full((batch_size,), sentiment_id, dtype=torch.long, device=device)
            
            # Use a simple dummy caption to get image embeddings
            dummy_captions = ["a photo"] * batch_size
            image_embs, _, _ = self.encoder(pil_images, dummy_captions, sentiment_ids)
            
            # Generate captions one by one
            generated_captions = []
            
            for i in range(batch_size):
                # Start with a simple word token instead of BOS
                tokenizer = self.decoder.tokenizer
                # Use "a" as starting token
                start_tokens = tokenizer("a", return_tensors="pt", add_special_tokens=False)
                input_ids = start_tokens['input_ids'].to(device)
                
                generated_ids = []
                
                for _ in range(max_length):
                    # Get text embeddings
                    text_embeddings = self.decoder.qwen_model.get_input_embeddings()(input_ids)
                    
                    # Add text modality embedding
                    text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=device)
                    text_mod_embs = self.encoder.modality_embedding(text_ids)
                    text_embeddings = text_embeddings + text_mod_embs
                    
                    # Add sentiment embedding
                    sentiment_emb = self.encoder.sentiment_embedding(sentiment_ids[i:i+1])
                    text_embeddings = text_embeddings + sentiment_emb.unsqueeze(1)
                    
                    # Get single image embedding
                    single_image_emb = image_embs[i:i+1]
                    
                    # Use the decoder's forward method
                    logits, _ = self.decoder(single_image_emb, text_embeddings, None, sentiment_ids[i:i+1])
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Top-k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                        next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                        next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                    else:
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    # Stop if EOS
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                    
                    generated_ids.append(next_token_id.item())
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)
                
                # Decode the generated sequence (skip the starting "a")
                if len(generated_ids) > 0:
                    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
                else:
                    caption = ""
                generated_captions.append(caption)
        
        return generated_captions
    
    def load_pretrained_weights(self, checkpoint_path):
        """
        Load weights from the pre-trained explained model.
        Only loads compatible weights, ignoring the new sentiment-related parameters.
        """
        print(f"Loading pre-trained weights from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # If the checkpoint contains 'encoder' and 'decoder' keys
            if 'encoder' in checkpoint and 'decoder' in checkpoint:
                # Load encoder weights (excluding sentiment_embedding)
                encoder_state = checkpoint['encoder']
                encoder_dict = self.encoder.state_dict()
                
                # Filter out sentiment_embedding from pre-trained weights
                filtered_encoder_state = {k: v for k, v in encoder_state.items() 
                                        if k in encoder_dict and 'sentiment_embedding' not in k}
                encoder_dict.update(filtered_encoder_state)
                self.encoder.load_state_dict(encoder_dict)
                
                # Load decoder weights (excluding sentiment_projection)
                decoder_state = checkpoint['decoder']
                decoder_dict = self.decoder.state_dict()
                
                # Filter out sentiment_projection from pre-trained weights
                filtered_decoder_state = {k: v for k, v in decoder_state.items() 
                                        if k in decoder_dict and 'sentiment_projection' not in k}
                decoder_dict.update(filtered_decoder_state)
                self.decoder.load_state_dict(decoder_dict)
                
                print("✅ Successfully loaded pre-trained weights (excluding new sentiment parameters)")
                
            else:
                print("⚠️  Checkpoint format not recognized. Expected 'encoder' and 'decoder' keys.")
                
        except Exception as e:
            print(f"❌ Error loading pre-trained weights: {e}")
            print("Continuing with randomly initialized weights...") 