from model import VisionLanguageEncoder as VisionLanguageEncoderBase, CaptionDecoder as CaptionDecoderBase
import torch
import torch.nn as nn

class VisionLanguageEncoder(VisionLanguageEncoderBase):
    """
    This encoder is largely the same as the original, but the forward pass is
    modified to return separate image and text embeddings. This is cleaner and
    prepares the data perfectly for our new cross-attention decoder.
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

        # 5. Add modality embeddings to distinguish between image and text
        image_ids = torch.zeros(image_patch_embeddings.shape[:2], dtype=torch.long, device=image_patch_embeddings.device)
        image_mod_embs = self.modality_embedding(image_ids)
        final_image_embeddings = image_patch_embeddings + image_mod_embs

        text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=text_embeddings.device)
        text_mod_embs = self.modality_embedding(text_ids)
        final_text_embeddings = text_embeddings + text_mod_embs

        return final_image_embeddings, final_text_embeddings, target_padded

class CaptionDecoder(CaptionDecoderBase):
    """
    The core of the improvement. This decoder uses cross-attention to intelligently
    fuse image and text information.
    
    Key Changes:
    - It takes separate image and text embeddings.
    - It uses the `vision_cross_attention` layer to let text "attend to" the image.
    - A LayerNorm is added for stability, a standard practice in transformers.
    - The loss is calculated cleanly on the text logits, as there's no more image prefix.
    """
    def __init__(self):
        super().__init__()
        # Adding a LayerNorm for stability after the attention operation
        self.attn_layer_norm = nn.LayerNorm(self.qwen_model.config.hidden_size)

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

        # 4. LOSS CALCULATION (if training)
        loss = None
        if target_tokens is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # The logits now directly correspond to the target tokens from start to end.
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
            
        return logits, loss 