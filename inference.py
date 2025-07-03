#!/usr/bin/env python3
"""
Simple inference script for the trained image captioning model.
Usage: python inference.py <path_to_image>
"""

import torch
import sys
from PIL import Image
from pathlib import Path
from model_explained import VisionLanguageEncoder, CaptionDecoder

class ImageCaptioner:
    def __init__(self, model_path="saved_models/explained/explained_20_epochs.pth"):
        """Initialize the image captioner with a trained model."""
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the model
        print("Loading model...")
        self.encoder = VisionLanguageEncoder().to(self.device)
        self.decoder = CaptionDecoder().to(self.device)
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        print("Model loaded successfully!")
    
    def generate_caption(self, image_path, max_length=30):
        """Generate a caption for the given image."""
        # Load and preprocess image
        image = self.load_image(image_path)
        
        with torch.no_grad():
            # Get image embeddings
            image_embeddings, _, _ = self.encoder([image], ["dummy"])
            
            # Start with SOS token
            sos_id = self.decoder.tokenizer.bos_token_id
            if sos_id is None:
                sos_id = self.decoder.tokenizer.eos_token_id
            
            input_ids = torch.tensor([[sos_id]], device=self.device)
            generated_tokens = []
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Get text embeddings for current sequence
                text_embeddings = self.decoder.qwen_model.get_input_embeddings()(input_ids)
                
                # Add modality embeddings
                text_ids = torch.ones_like(input_ids)
                text_mod_embs = self.encoder.modality_embedding(text_ids)
                text_embeddings = text_embeddings + text_mod_embs
                
                # Get next token prediction
                logits, _ = self.decoder(image_embeddings, text_embeddings)
                next_token_logits = logits[:, -1, :]  # Get last token's logits
                
                # Sample next token (using top-k sampling for better results)
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token_id = top_k_indices.gather(-1, next_token_idx)
                
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
        print("Usage: python inference.py <path_to_image>")
        print("Example: python inference.py data/images/sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found!")
        sys.exit(1)
    
    # Initialize captioner and generate caption
    try:
        captioner = ImageCaptioner()
        caption = captioner.generate_caption(image_path)
        
        print(f"\nImage: {image_path}")
        print(f"Caption: {caption}")
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 