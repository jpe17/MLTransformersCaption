from model import VisionLanguageEncoder as VisionLanguageEncoderBase, CaptionDecoder as CaptionDecoderBase
import torch
import torch.nn as nn
import math

class VisionLanguageEncoder(VisionLanguageEncoderBase):
    """
    This encoder is largely the same as the original, but the forward pass is
    modified to return separate image and text embeddings. This is cleaner and
    prepares the data perfectly for our new cross-attention decoder.
    """
    def __init__(self):
        super().__init__()
        # Initialize modality embeddings with smaller values for stability
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize image adapter with proper scaling
        for layer in self.image_adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Smaller gain for stability
                nn.init.constant_(layer.bias, 0.0)
    
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

        # 2. Adapt CLIP embeddings to Qwen's dimension with proper scaling
        image_patch_embeddings = self.image_adapter(image_patch_embeddings)
        
        # Apply L2 normalization to prevent extreme values
        image_patch_embeddings = torch.nn.functional.normalize(image_patch_embeddings, p=2, dim=-1)
        # Scale to match typical embedding magnitudes
        image_patch_embeddings = image_patch_embeddings * math.sqrt(self.qwen_model.config.hidden_size)

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

        # 5. Add modality embeddings to distinguish between image and text (with smaller scale)
        image_ids = torch.zeros(image_patch_embeddings.shape[:2], dtype=torch.long, device=device)
        image_mod_embs = self.modality_embedding(image_ids) * 0.1  # Scale down modality embeddings
        final_image_embeddings = image_patch_embeddings + image_mod_embs

        text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=device)
        text_mod_embs = self.modality_embedding(text_ids) * 0.1  # Scale down modality embeddings
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
        
        # Initialize cross-attention with proper scaling for stability
        hidden_size = self.qwen_model.config.hidden_size
        self.vision_cross_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1  # Add dropout for regularization
        )
        
        # Proper initialization of cross-attention weights
        for name, param in self.vision_cross_attention.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.1)  # Smaller gain for stability
                else:
                    nn.init.constant_(param, 0.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize layer norm properly
        nn.init.constant_(self.attn_layer_norm.weight, 1.0)
        nn.init.constant_(self.attn_layer_norm.bias, 0.0)
        
        # Add a scaling factor for the attention fusion
        self.attention_scale = nn.Parameter(torch.tensor(0.1))  # Start with small scale

    def forward(self, image_embeddings, text_embeddings, target_tokens=None):
        # image_embeddings: [batch_size, num_patches, hidden_size]
        # text_embeddings: [batch_size, seq_len, hidden_size]
        
        # Check for NaN/Inf in inputs
        if torch.isnan(image_embeddings).any() or torch.isinf(image_embeddings).any():
            print("Warning: NaN/Inf detected in image_embeddings")
            return None, None
        
        if torch.isnan(text_embeddings).any() or torch.isinf(text_embeddings).any():
            print("Warning: NaN/Inf detected in text_embeddings")
            return None, None
        
        # 1. CROSS-ATTENTION: Text embeddings (query) attend to image embeddings (key/value)
        # Apply layer norm to inputs for stability
        text_embeddings_norm = torch.nn.functional.layer_norm(text_embeddings, text_embeddings.shape[-1:])
        image_embeddings_norm = torch.nn.functional.layer_norm(image_embeddings, image_embeddings.shape[-1:])
        
        attn_output, attn_weights = self.vision_cross_attention(
            query=text_embeddings_norm,
            key=image_embeddings_norm,
            value=image_embeddings_norm
        )
        
        # Check for NaN/Inf in attention output
        if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
            print("Warning: NaN/Inf detected in attention output")
            return None, None

        # 2. FUSION: Combine text and attention output with a residual connection and normalization
        # Use learnable scaling and add residual connection
        scaled_attn = attn_output * torch.sigmoid(self.attention_scale)  # Ensure scale is in [0,1]
        fused_embeddings = self.attn_layer_norm(text_embeddings + scaled_attn)
        
        # Check for NaN/Inf after fusion
        if torch.isnan(fused_embeddings).any() or torch.isinf(fused_embeddings).any():
            print("Warning: NaN/Inf detected in fused_embeddings")
            return None, None

        # 3. DECODING: Pass the fused embeddings through the causal language model
        outputs = self.qwen_model(inputs_embeds=fused_embeddings)
        logits = outputs.logits
        
        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/Inf detected in logits")
            return None, None

        # 4. LOSS CALCULATION (if training)
        loss = None
        if target_tokens is not None:
            # Ensure target_tokens are on the same device
            target_tokens = target_tokens.to(logits.device)
            
            # Use proper loss function with padding token ignored
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,
                reduction='mean'  # Explicit reduction
            )
            
            # Shift logits and targets for causal language modeling
            # Make sure we have the right dimensions
            if logits.size(1) > target_tokens.size(1):
                # Trim logits to match target length
                logits = logits[:, :target_tokens.size(1), :]
            elif logits.size(1) < target_tokens.size(1):
                # Trim targets to match logit length
                target_tokens = target_tokens[:, :logits.size(1)]
            
            # Standard causal LM loss: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_tokens[..., 1:].contiguous()
            
            # Ensure shapes match
            if shift_logits.size(1) != shift_labels.size(1):
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]
            
            # Calculate loss
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf detected in loss calculation")
                return logits, None
            
        return logits, loss 