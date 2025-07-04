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
    def __init__(self, model_path=None):
        """Initialize the image captioner with a trained model."""
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # If no model path is provided, use a default or find the best one
        if model_path is None:
            default_model_path = "saved_models/explained/explained_20_epochs.pth"
            if Path(default_model_path).exists():
                print(f"No model path provided. Using default: {default_model_path}")
                model_path = default_model_path
            else:
                print(f"Default model '{default_model_path}' not found. Searching for best model in sweeps...")
                model_path = self.find_best_model()
                if model_path is None:
                    raise ValueError("No trained models found. Please provide a model_path or train a model first.")
        
        # Load the model
        print(f"Loading model from: {model_path}")
        self.encoder = VisionLanguageEncoder().to(self.device)
        self.decoder = CaptionDecoder().to(self.device)
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Print model info if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Model config: {config}")
        if 'final_train_loss' in checkpoint:
            print(f"Final training loss: {checkpoint['final_train_loss']:.4f}")
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        print("Model loaded successfully!")
    
    def find_best_model(self):
        """Find the best model from sweep runs based on lowest training loss."""
        sweep_dir = Path("saved_models/sweep_runs")
        if not sweep_dir.exists():
            return None
        
        best_model = None
        best_loss = float('inf')
        
        for model_file in sweep_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                if 'final_train_loss' in checkpoint:
                    loss = checkpoint['final_train_loss']
                    if loss < best_loss:
                        best_loss = loss
                        best_model = str(model_file)
            except Exception as e:
                print(f"Warning: Could not load {model_file}: {e}")
        
        if best_model:
            print(f"Found best model: {best_model} (loss: {best_loss:.4f})")
        
        return best_model
    
    @staticmethod
    def list_available_models():
        """List all available models from sweep runs."""
        sweep_dir = Path("saved_models/sweep_runs")
        if not sweep_dir.exists():
            print("No sweep models directory found.")
            return []
        
        models = []
        print("\nAvailable models from sweep runs:")
        print("-" * 80)
        
        for model_file in sweep_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                config = checkpoint.get('config', {})
                loss = checkpoint.get('final_train_loss', 'N/A')
                run_id = checkpoint.get('wandb_run_id', 'N/A')
                
                print(f"File: {model_file.name}")
                print(f"  Run ID: {run_id}")
                print(f"  Loss: {loss}")
                print(f"  Optimizer: {config.get('optimizer', 'N/A')}")
                print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
                print(f"  Scheduler: {config.get('scheduler', 'N/A')}")
                print("-" * 40)
                
                models.append({
                    'path': str(model_file),
                    'loss': loss,
                    'config': config,
                    'run_id': run_id
                })
            except Exception as e:
                print(f"Warning: Could not load {model_file}: {e}")
        
        return models
    
    def generate_caption(self, image_path, max_length=30):
        """Generate a caption for the given image."""
        # Load and preprocess image
        image = self.load_image(image_path)
        return self._generate_caption_from_pil(image, max_length)
    
    def generate_caption_from_pil(self, pil_image, max_length=30):
        """Generate a caption for a PIL Image object."""
        return self._generate_caption_from_pil(pil_image, max_length)
    
    def _generate_caption_from_pil(self, pil_image, max_length=30):
        """Internal method to generate caption from PIL Image."""
        with torch.no_grad():
            # Get image embeddings
            image_embeddings, _, _ = self.encoder([pil_image], ["dummy"])
            
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
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image> [model_path]")
        print("       python inference.py --list-models")
        print("Examples:")
        print("  python inference.py data/images/sample.jpg")
        print("  python inference.py data/images/sample.jpg saved_models/sweep_runs/model_sweep_abc123.pth")
        print("  python inference.py --list-models")
        sys.exit(1)
    
    # Handle list models command
    if sys.argv[1] == "--list-models":
        ImageCaptioner.list_available_models()
        sys.exit(0)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found!")
        sys.exit(1)
    
    # Initialize captioner and generate caption
    try:
        captioner = ImageCaptioner(model_path=model_path)
        caption = captioner.generate_caption(image_path)
        
        print(f"\nImage: {image_path}")
        print(f"Caption: {caption}")
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 