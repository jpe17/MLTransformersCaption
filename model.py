from loader import get_flickr_data
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


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
        
        # Create modality embedding layer (like positional embeddings in transformers)
        self.modality_embedding = torch.nn.Embedding(2, qwen_emb_dimension)  # 0=image, 1=text
    
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
        
        # Create modality IDs
        batch_size = image_patch_embeddings.shape[0]
        num_patches = image_patch_embeddings.shape[1]
        seq_len = input_embeddings.shape[1]
        
        # Image patches get ID 0, text tokens get ID 1
        image_ids = torch.zeros(batch_size, num_patches, dtype=torch.long)  # All 0s for image
        text_ids = torch.ones(batch_size, seq_len, dtype=torch.long)        # All 1s for text
        modality_ids = torch.cat([image_ids, text_ids], dim=1)              # [batch_size, total_seq_len]
        
        # Get modality embeddings and add to tokens
        mod_embeddings = self.modality_embedding(modality_ids)  # [batch_size, total_seq_len, hidden_size]
        
        # Concatenate image and text, then add modality embeddings
        combined_tokens = torch.cat([image_patch_embeddings, input_embeddings], dim=1)  # [batch_size, num_patches + seq_len, hidden_size]
        final_input = combined_tokens + mod_embeddings  # [batch_size, num_patches + seq_len, hidden_size]
        
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
            # The logits for the text part start after the image patches
            text_logits = logits[:, num_patches-1:-1, :]
            
            # Flatten for cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,
                label_smoothing=0.1
            )
            loss = loss_fct(text_logits.reshape(-1, text_logits.size(-1)), target_tokens.reshape(-1))
        
        return logits, loss


class TextOnlyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load Qwen3 base model - use AutoModelForCausalLM for compatibility
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, input_text, max_length=20):
        # Tokenize input with proper attention mask
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            # Generate with better parameters for base models
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Fix attention mask warning
                max_new_tokens=max_length,
                do_sample=True,  # Use sampling instead of greedy
                temperature=0.7,  # Add some randomness
                top_p=0.9,  # Nucleus sampling
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )
            
            # Extract only the new tokens (after the input)
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            
            # Decode the generated part
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text


def test_text_only_decoder():
    """Test the decoder with text-only input to see what it generates"""
    print("Testing Text-Only Decoder...")
    print("=" * 50)
    
    # Initialize the text-only decoder
    decoder = TextOnlyDecoder()
    decoder.eval()
    
    # Test prompts
    test_prompts = [
        "A beautiful",
        "The cat is",
        "In the garden",
        "This picture shows",
        "A person walking",
        "The weather is",
        "I can see",
        "The sky looks"
    ]
    
    with torch.no_grad():
        for prompt in test_prompts:
            generated = decoder(prompt, max_length=15)
            print(f"Input:  '{prompt}'")
            print(f"Output: '{prompt} {generated}'")
            print("-" * 30)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    test_text_only_decoder()
