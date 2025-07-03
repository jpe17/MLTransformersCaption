import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
import math


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load CLIP for vision encoding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, pil_images):
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt")
            vision_outputs = self.clip_model.vision_model(**clip_inputs)
            # Get patch embeddings (remove CLS token)
            image_patches = vision_outputs.last_hidden_state[:, 1:, :]  # [batch_size, 49, 768]
        return image_patches


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to vision
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, vision_features, causal_mask=None):
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        
        # Cross-attention to vision
        cross_out, _ = self.cross_attn(x, vision_features, vision_features)
        x = self.norm2(x + cross_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        
        return x


class CustomDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.d_model = d_model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Vision projection
        self.vision_proj = nn.Linear(768, d_model)
        
        # Transformer blocks with self + cross attention
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Output with dropout
        self.dropout = nn.Dropout(0.1)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, image_features, target_tokens=None, max_length=20):
        batch_size = image_features.shape[0]
        
        # Project vision features
        vision_features = self.vision_proj(image_features)  # [batch_size, 49, d_model]
        
        if target_tokens is not None:
            # Training mode
            seq_len = target_tokens.shape[1]
            
            # Embed tokens
            x = self.token_embedding(target_tokens) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(target_tokens.device)
            causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
            
            # Pass through transformer blocks
            for layer in self.layers:
                x = layer(x, vision_features, causal_mask)
            
            # Final layer norm and projection
            x = self.dropout(x)
            x = self.ln_f(x)
            logits = self.output_proj(x)
            return logits
        
        else:
            # Inference mode
            sos_token = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            generated = torch.full((batch_size, 1), sos_token, device=image_features.device)
            
            for _ in range(max_length):
                seq_len = generated.shape[1]
                
                # Embed current sequence
                x = self.token_embedding(generated) * math.sqrt(self.d_model)
                x = self.pos_encoding(x)
                
                # Create causal mask
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                causal_mask = causal_mask.to(generated.device)
                causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
                
                # Pass through transformer blocks
                for layer in self.layers:
                    x = layer(x, vision_features, causal_mask)
                
                # Get next token (no dropout during inference)
                x = self.ln_f(x)
                logits = self.output_proj(x)
                
                # Add temperature sampling to avoid always picking most likely token
                temperature = 0.8
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token], dim=1)
            
            return generated[:, 1:]


class ImageCaptionModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.vision_encoder = VisionEncoder()
        
        # Get vocab size from tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        vocab_size = len(tokenizer)
        
        self.decoder = CustomDecoder(vocab_size, d_model, nhead, num_layers)
        self.tokenizer = tokenizer
    
    def forward(self, pil_images, captions=None):
        # Encode images
        image_features = self.vision_encoder(pil_images)
        
        if captions is not None:
            # Training mode
            # Tokenize captions
            tokenized = self.tokenizer(captions, padding=True, truncation=True, 
                                     return_tensors="pt", add_special_tokens=True)
            target_tokens = tokenized['input_ids']
            
            # Get logits
            logits = self.decoder(image_features, target_tokens)
            
            # Calculate loss with label smoothing to prevent overconfidence
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, label_smoothing=0.1)
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_tokens[..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return logits, loss
        
        else:
            # Inference mode
            generated_tokens = self.decoder(image_features)
            captions = [self.tokenizer.decode(tokens, skip_special_tokens=True) 
                       for tokens in generated_tokens]
            return captions


# Test the model
if __name__ == "__main__":
    from loader import get_flickr_data
    
    # Load data
    train, val = get_flickr_data()
    train_batches = train()
    
    # Test model
    model = ImageCaptionModel(d_model=256, nhead=8, num_layers=4)  # Smaller for speed
    
    for pil_images, captions in train_batches:
        print("Testing custom model...")
        print("Images:", len(pil_images))
        print("Captions:", captions[:2])
        
        # Training forward pass
        logits, loss = model(pil_images, captions)
        print("Logits shape:", logits.shape)
        print("Loss:", loss.item())
        
        # Inference forward pass
        generated_captions = model(pil_images[:2])  # Test on first 2 images
        print("Generated:", generated_captions)
        
        break 