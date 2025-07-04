from model_explained import VisionLanguageEncoder, CaptionDecoder
from PIL import Image
import torch
import os

class SimpleDualCaptionSystem:
    """Simple dual caption system using two separate models"""
    
    def __init__(self, pos_model_path="saved_models/simple_dual/positive_model.pth",
                 neg_model_path="saved_models/simple_dual/negative_model.pth"):
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load positive model
        self.pos_encoder = VisionLanguageEncoder().to(self.device)
        self.pos_decoder = CaptionDecoder().to(self.device)
        
        if os.path.exists(pos_model_path):
            checkpoint = torch.load(pos_model_path, map_location=self.device)
            self.pos_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.pos_decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print("✅ Positive model loaded")
        else:
            print(f"❌ Positive model not found: {pos_model_path}")
            
        # Load negative model
        self.neg_encoder = VisionLanguageEncoder().to(self.device)
        self.neg_decoder = CaptionDecoder().to(self.device)
        
        if os.path.exists(neg_model_path):
            checkpoint = torch.load(neg_model_path, map_location=self.device)
            self.neg_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.neg_decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print("✅ Negative model loaded")
        else:
            print(f"❌ Negative model not found: {neg_model_path}")
        
        # Set to eval mode
        self.pos_encoder.eval()
        self.pos_decoder.eval()
        self.neg_encoder.eval()
        self.neg_decoder.eval()
    
    def generate_caption(self, image, model_type="positive", max_length=20):
        """Generate caption using either positive or negative model"""
        
        # Choose the right model
        if model_type == "positive":
            encoder, decoder = self.pos_encoder, self.pos_decoder
        else:
            encoder, decoder = self.neg_encoder, self.neg_decoder
        
        with torch.no_grad():
            # Preprocess image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            image = image.resize((224, 224))
            
            # Get image embeddings
            image_embeddings, _, _ = encoder([image], ["a photo"])
            
            # Generate caption
            tokenizer = decoder.tokenizer
            start_tokens = tokenizer("a", return_tensors="pt", add_special_tokens=False)
            input_ids = start_tokens['input_ids'].to(self.device)
            
            generated_ids = []
            
            for _ in range(max_length):
                # Get text embeddings
                text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
                
                # Add text modality embedding
                text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=self.device)
                text_mod_embs = encoder.modality_embedding(text_ids)
                text_embeddings = text_embeddings + text_mod_embs
                
                # Get logits
                logits, _ = decoder(image_embeddings, text_embeddings)
                next_token_logits = logits[:, -1, :] / 0.8
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, 40)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_relative_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
                
                # Stop if EOS
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Decode
            if len(generated_ids) > 0:
                caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                caption = ""
            
            return caption
    
    def generate_both_captions(self, image, max_length=20):
        """Generate both positive and negative captions"""
        pos_caption = self.generate_caption(image, "positive", max_length)
        neg_caption = self.generate_caption(image, "negative", max_length)
        return pos_caption, neg_caption

def main():
    print("🎯 Simple Dual Caption System")
    print("=" * 40)
    
    # Initialize the system
    system = SimpleDualCaptionSystem()
    
    # Test images
    test_images = [
        "data/images/1000268201_693b08cb0e.jpg",
        "data/images/1001773457_577c3a7d70.jpg",
        "data/images/1002674143_1b742ab4b8.jpg"
    ]
    
    for i, image_path in enumerate(test_images):
        if os.path.exists(image_path):
            print(f"\n📸 Image {i+1}: {os.path.basename(image_path)}")
            print("-" * 30)
            
            pos_caption, neg_caption = system.generate_both_captions(image_path)
            
            print(f"😊 Positive: {pos_caption}")
            print(f"😠 Negative: {neg_caption}")
        else:
            print(f"⚠️  Image not found: {image_path}")

if __name__ == "__main__":
    main() 