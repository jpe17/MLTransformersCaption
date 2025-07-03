from model import VisionLanguageEncoder as VisionLanguageEncoderBase, CaptionDecoder as CaptionDecoderBase
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionLanguageEncoder(VisionLanguageEncoderBase):
    """
    Same as the base encoder but returns separate components for the new embedding-based loss.
    """
    def forward(self, pil_images, captions):
        # 1. Process images with CLIP to get patch embeddings
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.qwen_model.device)
            vision_outputs = self.clip_model.vision_model(**clip_inputs)
            image_patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # Exclude CLS token

        # 2. Adapt CLIP embeddings to Qwen's dimension
        image_patch_embeddings = self.image_adapter(image_patch_embeddings)

        # 3. Process text captions to get input and target tokens
        tokenized = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).to(self.qwen_model.device)
        tokens = tokenized['input_ids']
        
        sos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        eos = self.tokenizer.eos_token_id
        
        sos_tokens = torch.full((tokens.shape[0], 1), sos, dtype=tokens.dtype, device=tokens.device)
        input_padded = torch.cat([sos_tokens, tokens], dim=1)
        
        eos_tokens = torch.full((tokens.shape[0], 1), eos, dtype=tokens.dtype, device=tokens.device)
        target_padded = torch.cat([tokens, eos_tokens], dim=1)

        # 4. Get text embeddings from Qwen's embedding layer
        with torch.no_grad():
            text_embeddings = self.qwen_model.get_input_embeddings()(input_padded)
            # Also get target embeddings for the new loss
            target_embeddings = self.qwen_model.get_input_embeddings()(target_padded)

        # 5. Add modality embeddings
        image_ids = torch.zeros(image_patch_embeddings.shape[:2], dtype=torch.long, device=image_patch_embeddings.device)
        image_mod_embs = self.modality_embedding(image_ids)
        final_image_embeddings = image_patch_embeddings + image_mod_embs

        text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=text_embeddings.device)
        text_mod_embs = self.modality_embedding(text_ids)
        final_text_embeddings = text_embeddings + text_mod_embs

        return final_image_embeddings, final_text_embeddings, target_padded, target_embeddings

class CaptionDecoder(CaptionDecoderBase):
    """
    Enhanced decoder that uses embedding-based loss instead of cross-entropy.
    
    Key Innovation: Instead of comparing predicted token IDs to target token IDs,
    we compare the predicted embeddings to target embeddings using cosine similarity.
    This allows the model to learn semantic relationships rather than exact matches.
    """
    def __init__(self):
        super().__init__()
        
        # Add components for embedding-based loss
        self.embedding_projection = nn.Linear(
            self.qwen_model.config.hidden_size, 
            self.qwen_model.config.hidden_size
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Layer norm for stability
        self.embedding_norm = nn.LayerNorm(self.qwen_model.config.hidden_size)

    def forward(self, image_embeddings, text_embeddings, target_tokens=None, target_embeddings=None):
        # image_embeddings: [batch_size, num_patches, hidden_size]
        # text_embeddings: [batch_size, seq_len, hidden_size]
        
        # 1. Cross-attention between text and image
        attn_output, _ = self.vision_cross_attention(
            query=text_embeddings,
            key=image_embeddings,
            value=image_embeddings
        )

        # 2. Fusion with residual connection
        fused_embeddings = text_embeddings + attn_output
        fused_embeddings = self.embedding_norm(fused_embeddings)

        # 3. Pass through language model
        outputs = self.qwen_model(inputs_embeds=fused_embeddings)
        logits = outputs.logits
        
        # 4. Get predicted embeddings by projecting logits back to embedding space
        # This is the key innovation: instead of using logits directly, we convert them
        # to a "predicted embedding" that we can compare with target embeddings
        predicted_embeddings = self.get_predicted_embeddings(logits)

        # 5. Calculate embedding-based loss
        loss = None
        if target_tokens is not None and target_embeddings is not None:
            loss = self.embedding_loss(predicted_embeddings, target_embeddings, target_tokens)
            
        return logits, loss, predicted_embeddings

    def get_predicted_embeddings(self, logits):
        """
        Convert logits to predicted embeddings using a weighted combination of the vocabulary embeddings.
        
        Instead of taking argmax (which is non-differentiable), we use the softmax probabilities
        as weights to create a "soft" embedding that represents what the model is predicting.
        """
        # Get softmax probabilities from logits
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # Get the embedding matrix
        embedding_matrix = self.qwen_model.get_input_embeddings().weight  # [vocab_size, hidden_size]
        
        # Compute weighted sum: for each position, sum embeddings weighted by their probabilities
        predicted_embeddings = torch.matmul(probs, embedding_matrix)  # [batch_size, seq_len, hidden_size]
        
        # Apply projection and normalization
        predicted_embeddings = self.embedding_projection(predicted_embeddings)
        predicted_embeddings = self.embedding_norm(predicted_embeddings)
        
        return predicted_embeddings

    def embedding_loss(self, predicted_embeddings, target_embeddings, target_tokens):
        """
        Calculate loss based on embedding similarity rather than exact token matching.
        
        Uses a combination of:
        1. Cosine similarity loss (main semantic similarity)
        2. MSE loss (for magnitude matching)
        3. Contrastive loss (to distinguish between different tokens)
        """
        # Mask out padding tokens
        mask = (target_tokens != self.tokenizer.pad_token_id).float()
        
        # 1. Cosine similarity loss - encourages semantic alignment
        cosine_sim = F.cosine_similarity(predicted_embeddings, target_embeddings, dim=-1)
        cosine_loss = (1 - cosine_sim) * mask
        cosine_loss = cosine_loss.sum() / mask.sum()
        
        # 2. MSE loss - encourages magnitude matching
        mse_loss = F.mse_loss(predicted_embeddings, target_embeddings, reduction='none').mean(dim=-1)
        mse_loss = (mse_loss * mask).sum() / mask.sum()
        
        # 3. Contrastive loss - encourages distinguishing between different tokens
        # This helps prevent the model from collapsing to similar embeddings
        contrastive_loss = self.contrastive_loss(predicted_embeddings, target_embeddings, mask)
        
        # Combine losses with weights
        total_loss = 0.5 * cosine_loss + 0.3 * mse_loss + 0.2 * contrastive_loss
        
        return total_loss

    def contrastive_loss(self, predicted_embeddings, target_embeddings, mask):
        """
        Contrastive loss to ensure the model learns to distinguish between different tokens.
        """
        batch_size, seq_len, hidden_size = predicted_embeddings.shape
        
        # Flatten embeddings for easier computation
        pred_flat = predicted_embeddings.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        target_flat = target_embeddings.view(-1, hidden_size)
        mask_flat = mask.view(-1)
        
        # Only compute loss for non-padded tokens
        valid_indices = mask_flat.bool()
        if valid_indices.sum() == 0:
            return torch.tensor(0.0, device=predicted_embeddings.device)
        
        pred_valid = pred_flat[valid_indices]
        target_valid = target_flat[valid_indices]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(pred_valid, target_valid.t()) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(len(pred_valid), device=predicted_embeddings.device)
        
        # Contrastive loss (InfoNCE style)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class EmbeddingLossTrainer:
    """
    Helper class to manage the training process with embedding-based loss.
    """
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def train_step(self, pil_images, captions):
        """
        Single training step with embedding-based loss.
        """
        # Get embeddings from encoder
        image_embeddings, text_embeddings, target_tokens, target_embeddings = self.encoder(pil_images, captions)
        
        # Forward pass through decoder
        logits, loss, predicted_embeddings = self.decoder(
            image_embeddings, text_embeddings, target_tokens, target_embeddings
        )
        
        return loss, logits, predicted_embeddings
    
    def generate_caption(self, pil_image, max_length=30):
        """
        Generate caption using the embedding-based model.
        """
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Process image
            image_embeddings, _, _, _ = self.encoder([pil_image], ["dummy"])
            
            # Start with SOS token
            sos_id = self.decoder.tokenizer.bos_token_id or self.decoder.tokenizer.eos_token_id
            input_ids = torch.tensor([[sos_id]], device=self.device)
            
            generated_ids = []
            
            for _ in range(max_length):
                # Get current text embeddings
                text_embeddings = self.decoder.qwen_model.get_input_embeddings()(input_ids)
                
                # Add modality embeddings
                text_mod_ids = torch.ones_like(input_ids)
                text_mod_embs = self.encoder.modality_embedding(text_mod_ids)
                text_embeddings = text_embeddings + text_mod_embs
                
                # Get logits
                logits, _, _ = self.decoder(image_embeddings, text_embeddings)
                
                # Sample next token
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                if next_token_id.item() == self.decoder.tokenizer.eos_token_id:
                    break
                    
                generated_ids.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            return self.decoder.tokenizer.decode(generated_ids, skip_special_tokens=True) 