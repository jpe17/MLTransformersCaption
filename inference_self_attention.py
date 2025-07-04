#!/usr/bin/env python3
"""
Inference script for the self-attention image captioning model.
Usage: python inference_self_attention.py <path_to_image>
"""

import torch
import sys
from PIL import Image
from pathlib import Path
from model_self_attention import VisionLanguageEncoder, CaptionDecoder

class SelfAttentionImageCaptioner:
    def __init__(self, model_path="saved_models/self_attention/self_attention_10_epochs.pth"):
        """Initialize the image captioner with a trained self-attention model."""
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the model
        print("Loading self-attention model...")
        self.encoder = VisionLanguageEncoder().to(self.device)
        self.decoder = CaptionDecoder().to(self.device)
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        print("Self-attention model loaded successfully!")
    
    def generate_caption(self, image_path, max_length=30):
        """Generate a caption for the given image using self-attention."""
        # Load and preprocess image
        image = self.load_image(image_path)
        
        with torch.no_grad():
            # Get image embeddings through encoder (using dummy caption)
            combined_embeddings, _, num_patches = self.encoder([image], ["dummy caption"])
            
            # Extract just the image part for generation
            image_embeddings = combined_embeddings[:, :num_patches, :]
            
            # Start with SOS token
            sos_id = self.decoder.tokenizer.bos_token_id
            if sos_id is None:
                sos_id = self.decoder.tokenizer.eos_token_id
            
            input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=self.device)
            generated_tokens = []
            
            # Generate tokens one by one using self-attention approach
            for _ in range(max_length):
                # Get text embeddings for current sequence
                text_embeddings = self.decoder.qwen_model.get_input_embeddings()(input_ids)
                
                # Add modality embeddings for text tokens
                text_mod_id = torch.ones_like(input_ids)
                text_mod_emb = self.encoder.modality_embedding(text_mod_id)
                text_embeddings_final = text_embeddings + text_mod_emb
                
                # Combine image and text embeddings
                combined_for_generation = torch.cat([image_embeddings, text_embeddings_final], dim=1)
                
                # Add enhanced positional embeddings (key difference from explained model)
                batch_size, seq_len, hidden_size = combined_for_generation.shape
                position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                pos_embeddings = self.decoder.enhanced_pos_embedding(position_ids)
                enhanced_embeddings = combined_for_generation + pos_embeddings
                
                # Apply layer normalization (key difference from explained model)
                enhanced_embeddings = self.decoder.input_layer_norm(enhanced_embeddings)
                
                # Get next token prediction through self-attention
                with torch.amp.autocast(device_type=self.device, enabled=(self.device == 'cuda' or self.device == 'mps')):
                    outputs = self.decoder.qwen_model(inputs_embeds=enhanced_embeddings)
                    logits = outputs.logits
                
                next_token_logits = logits[:, -1, :]  # Get last token's logits
                
                # Sample next token using top-k sampling
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                
                # Check for end of sequence
                if next_token_id.item() == self.decoder.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Decode the generated tokens
            caption = self.decoder.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return caption.strip()
    
    def load_image(self, image_path):
        """Load and preprocess an image from any format."""
        try:
            image = Image.open(image_path)
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Could not load image from {image_path}: {e}")

def main():
    """Main function for command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python inference_self_attention.py <path_to_image>")
        print("Example: python inference_self_attention.py data/images/sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found!")
        sys.exit(1)
    
    # Initialize captioner and generate caption
    try:
        captioner = SelfAttentionImageCaptioner()
        caption = captioner.generate_caption(image_path)
        
        print(f"\nImage: {image_path}")
        print(f"Caption: {caption}")
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 