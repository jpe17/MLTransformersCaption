#!/usr/bin/env python3
"""
Emergency save script - saves only the trained parameters (much smaller file)
Run this immediately after training to save the essential weights.
"""

import torch
from pathlib import Path

def emergency_save_trained_params(encoder, decoder, save_path="emergency_trained_params.pth"):
    """
    Save only the trained parameters - much smaller than full model
    """
    print(f"Saving trained parameters to {save_path}...")
    
    # Only save the parameters that were actually trained
    trained_params = {
        # Encoder trained parts
        'encoder_image_adapter': encoder.image_adapter.state_dict(),
        'encoder_modality_embedding': encoder.modality_embedding.state_dict(),
        
        # Decoder trained parts  
        'decoder_enhanced_pos_embedding': decoder.enhanced_pos_embedding.state_dict(),
        'decoder_input_layer_norm': decoder.input_layer_norm.state_dict(),
        'decoder_last_2_layers': {
            f'layer_{i}': layer.state_dict() 
            for i, layer in enumerate(decoder.qwen_model.model.layers[-2:])
        }
    }
    
    # Save with compression to make it even smaller
    torch.save(trained_params, save_path)
    
    file_size = Path(save_path).stat().st_size / (1024*1024)  # MB
    print(f"✅ Saved! File size: {file_size:.1f} MB")
    print(f"This is much smaller than the full model (~5GB)")

if __name__ == "__main__":
    print("This script should be run from your training code")
    print("Add these lines to your training script after line 93:")
    print()
    print("# Emergency save before full save")
    print("from emergency_save import emergency_save_trained_params")
    print("emergency_save_trained_params(encoder, decoder)")
    print()
    print("Then run your training script again - it will save the small file first!") 