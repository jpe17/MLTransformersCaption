#!/usr/bin/env python3
"""
Test script to validate model stability before running sweeps.
This will help catch NaN/Inf issues early.
"""

import torch
from model_explained import VisionLanguageEncoder, CaptionDecoder
from loader import get_flickr_data
import numpy as np
from PIL import Image

def test_model_stability():
    print("Testing model stability...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create models
    encoder = VisionLanguageEncoder().to(device)
    decoder = CaptionDecoder().to(device)
    
    # Set to training mode
    encoder.train()
    decoder.train()
    
    # Create dummy data
    dummy_image = Image.new('RGB', (224, 224), color='red')
    dummy_caption = "A red image for testing"
    
    print("Testing forward pass...")
    
    try:
        # Test encoder
        image_embeddings, text_embeddings, target_tokens = encoder([dummy_image], [dummy_caption])
        
        print(f"Image embeddings shape: {image_embeddings.shape}")
        print(f"Text embeddings shape: {text_embeddings.shape}")
        print(f"Target tokens shape: {target_tokens.shape}")
        
        # Check for NaN/Inf in encoder outputs
        if torch.isnan(image_embeddings).any() or torch.isinf(image_embeddings).any():
            print("❌ FAIL: NaN/Inf detected in image embeddings")
            return False
        
        if torch.isnan(text_embeddings).any() or torch.isinf(text_embeddings).any():
            print("❌ FAIL: NaN/Inf detected in text embeddings")
            return False
        
        print("✅ Encoder outputs are stable")
        
        # Test decoder
        logits, loss = decoder(image_embeddings, text_embeddings, target_tokens)
        
        if logits is None or loss is None:
            print("❌ FAIL: Decoder returned None values")
            return False
        
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss.item()}")
        
        # Check for NaN/Inf in decoder outputs
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("❌ FAIL: NaN/Inf detected in logits")
            return False
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ FAIL: NaN/Inf detected in loss")
            return False
        
        print("✅ Decoder outputs are stable")
        
        # Test backward pass
        print("Testing backward pass...")
        loss.backward()
        
        # Check gradients
        has_nan_grad = False
        trainable_params = list(encoder.image_adapter.parameters()) + \
                          list(encoder.modality_embedding.parameters()) + \
                          list(decoder.vision_cross_attention.parameters()) + \
                          list(decoder.attn_layer_norm.parameters()) + \
                          [decoder.attention_scale]
        
        for i, param in enumerate(trainable_params):
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"❌ FAIL: NaN/Inf gradients detected in parameter {i}")
                    has_nan_grad = True
                    break
        
        if has_nan_grad:
            return False
        
        print("✅ Gradients are stable")
        
        # Test with real data
        print("Testing with real data...")
        train_loader_fn, _ = get_flickr_data()
        train_batches = train_loader_fn()
        
        for i, (pil_images, captions) in enumerate(train_batches):
            if i >= 3:  # Test only first 3 batches
                break
            
            print(f"Testing batch {i+1}...")
            
            # Forward pass
            image_embeddings, text_embeddings, target_tokens = encoder(pil_images, captions)
            
            if image_embeddings is None or text_embeddings is None or target_tokens is None:
                print(f"❌ FAIL: Encoder returned None for batch {i+1}")
                return False
            
            logits, loss = decoder(image_embeddings, text_embeddings, target_tokens)
            
            if logits is None or loss is None:
                print(f"❌ FAIL: Decoder returned None for batch {i+1}")
                return False
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ FAIL: NaN/Inf loss in batch {i+1}: {loss.item()}")
                return False
            
            print(f"  Batch {i+1} loss: {loss.item():.4f}")
        
        print("✅ All tests passed! Model is stable.")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception during testing: {e}")
        return False

def test_attention_scale():
    """Test that attention scale is properly initialized and bounded"""
    print("\nTesting attention scale parameter...")
    
    decoder = CaptionDecoder()
    
    # Check initial value
    scale_value = decoder.attention_scale.item()
    print(f"Initial attention scale: {scale_value}")
    
    # Check that sigmoid keeps it in [0,1]
    sigmoid_scale = torch.sigmoid(decoder.attention_scale).item()
    print(f"Sigmoid attention scale: {sigmoid_scale}")
    
    if 0 <= sigmoid_scale <= 1:
        print("✅ Attention scale is properly bounded")
        return True
    else:
        print("❌ FAIL: Attention scale is not properly bounded")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL STABILITY TEST")
    print("=" * 60)
    
    # Test attention scale
    if not test_attention_scale():
        exit(1)
    
    # Test model stability
    if not test_model_stability():
        print("\n❌ Model stability test FAILED!")
        print("Please fix the issues before running sweeps.")
        exit(1)
    
    print("\n🎉 All tests passed! Model is ready for training.") 