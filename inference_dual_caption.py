from model_dual_caption import DualCaptionModel
from PIL import Image
import torch
import os

def load_dual_caption_model(model_path="saved_models/dual_caption/dual_caption_model.pth"):
    """Load the trained dual caption model"""
    model = DualCaptionModel()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"❌ Model not found at {model_path}")
        return None
    
    model.eval()
    return model

def generate_dual_captions(model, image_path, max_length=20):
    """Generate both positive and negative captions for an image"""
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Resize to match training
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
    # Generate captions
    with torch.no_grad():
        pos_caption = model.generate_caption(
            [image], 
            sentiment_type="positive", 
            max_length=max_length,
            temperature=0.8,
            top_k=40
        )[0]
        
        neg_caption = model.generate_caption(
            [image], 
            sentiment_type="negative", 
            max_length=max_length,
            temperature=0.8,
            top_k=40
        )[0]
    
    return pos_caption, neg_caption

def main():
    print("🎯 Dual Caption Model Inference")
    print("=" * 40)
    
    # Load model
    model = load_dual_caption_model()
    if model is None:
        return
    
    # Test with some images
    test_images = [
        "data/images/1000268201_693b08cb0e.jpg",
        "data/images/1001773457_577c3a7d70.jpg",
        "data/images/1002674143_1b742ab4b8.jpg"
    ]
    
    for i, image_path in enumerate(test_images):
        if os.path.exists(image_path):
            print(f"\n📸 Image {i+1}: {os.path.basename(image_path)}")
            print("-" * 30)
            
            pos_caption, neg_caption = generate_dual_captions(model, image_path)
            
            print(f"😊 Positive: {pos_caption}")
            print(f"😠 Negative: {neg_caption}")
        else:
            print(f"⚠️  Image not found: {image_path}")

if __name__ == "__main__":
    main() 