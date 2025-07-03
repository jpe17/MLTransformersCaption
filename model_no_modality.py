from loader import get_flickr_data
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from torch.nn.utils.rnn import pad_sequence


class VisionLanguageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the Qwen tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        self.qwen_model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B-Base")
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        # Simple modality embeddings for Qwen compatibility
        qwen_emb_dimension = self.qwen_model.config.hidden_size
        
        # Better alignment: make CLIP embeddings more similar to Qwen text embeddings
        self.image_adapter = torch.nn.Sequential(
            torch.nn.Linear(768, qwen_emb_dimension),
            torch.nn.LayerNorm(qwen_emb_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(qwen_emb_dimension, qwen_emb_dimension),
            torch.nn.LayerNorm(qwen_emb_dimension)
        ) 
        
    def forward(self, pil_images, captions):
        # pil_images: list of PIL Images (224x224)
        # captions: list of captions for this batch
        
        # Process images with CLIP (get patch embeddings, not pooled)
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt")
            # Get vision encoder outputs (patch embeddings)
            vision_outputs = self.clip_model.vision_model(**clip_inputs)
            image_patch_embeddings = vision_outputs.last_hidden_state  # Shape: [batch_size, num_patches+1, 768]
            
            # Remove the CLS token (first token) to get only patch embeddings
            image_patch_embeddings = image_patch_embeddings[:, 1:, :]  # Shape: [batch_size, num_patches, 768]
        
        # Project and align CLIP embeddings to be more Qwen-like
        image_patch_embeddings = self.image_adapter(image_patch_embeddings)  # Shape: [batch_size, num_patches, qwen_dim]
        
        # Tokenize all captions at once
        tokenized = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        tokens = tokenized['input_ids']  # Shape: [batch_size, max_length]
        
        # Get special tokens
        sos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        eos = self.tokenizer.eos_token_id
        
        # Create input tokens: [SOS, token1, token2, ...]
        sos_tokens = torch.full((tokens.shape[0], 1), sos, dtype=tokens.dtype)
        input_padded = torch.cat([sos_tokens, tokens], dim=1)  # Shape: [batch_size, seq_len+1]
        
        # Create target tokens: [token1, token2, ..., EOS]
        eos_tokens = torch.full((tokens.shape[0], 1), eos, dtype=tokens.dtype)
        target_padded = torch.cat([tokens, eos_tokens], dim=1)  # Shape: [batch_size, seq_len+1]
        
        # Get embeddings from the model
        with torch.no_grad():
            # Get the embedding layer from the model
            embeddings = self.qwen_model.get_input_embeddings()
            
            # Convert token IDs to embeddings
            input_embeddings = embeddings(input_padded)  # Shape: [batch_size, seq_len, hidden_size]
        
        num_patches = image_patch_embeddings.shape[1]
        
        # Concatenate image and text
        final_input = torch.cat([image_patch_embeddings, input_embeddings], dim=1)  # [batch_size, num_patches + seq_len, hidden_size]
        
        return final_input, target_padded, num_patches


class CaptionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load Qwen3 base model - use AutoModelForCausalLM like the working TextOnlyDecoder
        from transformers import AutoModelForCausalLM
        self.qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze most of Qwen, but unfreeze last 2 layers for adaptation
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 transformer layers (note: model.model.layers for AutoModelForCausalLM)
        if hasattr(self.qwen_model.model, 'layers'):
            for layer in self.qwen_model.model.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Get model dimensions
        hidden_size = self.qwen_model.config.hidden_size
        
        # Cross-attention to let text attend to image patches
        self.vision_cross_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, combined_input, target_tokens=None, num_patches=None):
        # combined_input: [batch_size, num_patches + seq_len, hidden_size]
        # target_tokens: [batch_size, target_seq_len] - for training
        
        # Pass through Qwen model with embeddings input (like the working approach)
        outputs = self.qwen_model(inputs_embeds=combined_input)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size] - AutoModelForCausalLM has built-in logits
        
        loss = None
        if target_tokens is not None and num_patches is not None:
            # Calculate loss only on text tokens (skip image patches)
            text_logits = logits[:, num_patches:, :]  # [batch_size, text_seq_len, vocab_size]
            
            # Flatten for cross entropy loss
            shift_logits = text_logits[..., :-1, :].contiguous()  # [batch_size, text_seq_len-1, vocab_size]
            shift_labels = target_tokens[..., 1:].contiguous()    # [batch_size, text_seq_len-1]
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return logits, loss 