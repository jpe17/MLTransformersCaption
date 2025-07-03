from model import VisionLanguageEncoder as VisionLanguageEncoderBase, CaptionDecoder as CaptionDecoderBase
import torch
import torch.nn as nn

class VisionLanguageEncoder(VisionLanguageEncoderBase):
    """
    Self-attention only encoder that prepares data cleanly like the explained model.
    Key insight: Instead of cross-attention, we'll use better positional and modality
    embeddings to help self-attention distinguish between image and text tokens.
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

        # 5. Enhanced modality embeddings - make them stronger for self-attention
        image_ids = torch.zeros(image_patch_embeddings.shape[:2], dtype=torch.long, device=image_patch_embeddings.device)
        image_mod_embs = self.modality_embedding(image_ids)
        final_image_embeddings = image_patch_embeddings + image_mod_embs

        text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=text_embeddings.device)
        text_mod_embs = self.modality_embedding(text_ids)
        final_text_embeddings = text_embeddings + text_mod_embs

        # 6. Concatenate for self-attention (image first, then text)
        combined_embeddings = torch.cat([final_image_embeddings, final_text_embeddings], dim=1)
        num_patches = final_image_embeddings.shape[1]

        return combined_embeddings, target_padded, num_patches

class CaptionDecoder(CaptionDecoderBase):
    """
    Self-attention only decoder that uses the successful patterns from the explained model.
    
    Key improvements:
    - Enhanced positional embeddings to help distinguish image vs text positions
    - Layer normalization for stability
    - Better loss calculation alignment
    - Proper gradient flow through unfrozen layers
    """
    def __init__(self):
        super().__init__()
        # Add enhanced positional embeddings for better position awareness
        hidden_size = self.qwen_model.config.hidden_size
        self.enhanced_pos_embedding = nn.Embedding(512, hidden_size)  # Support up to 512 tokens
        
        # Add layer norm for stability (inspired by the explained model)
        self.input_layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize positional embeddings
        nn.init.normal_(self.enhanced_pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, combined_embeddings, target_tokens=None, num_patches=None):
        batch_size, seq_len, hidden_size = combined_embeddings.shape
        
        # 1. Add enhanced positional embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=combined_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.enhanced_pos_embedding(position_ids)
        
        # 2. Combine with positional information and normalize
        enhanced_embeddings = combined_embeddings + pos_embeddings
        enhanced_embeddings = self.input_layer_norm(enhanced_embeddings)

        # 3. Pass through Qwen model with self-attention
        outputs = self.qwen_model(inputs_embeds=enhanced_embeddings)
        logits = outputs.logits

        # 4. Clean loss calculation (inspired by explained model)
        loss = None
        if target_tokens is not None and num_patches is not None:
            # Extract logits for text tokens only (skip image patches)
            text_logits = logits[:, num_patches:num_patches+target_tokens.shape[1], :]
            
            # Direct alignment between logits and target tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), target_tokens.view(-1))
            
        return logits, loss, num_patches 