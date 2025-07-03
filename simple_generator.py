#!/usr/bin/env python3
"""
Simplified synthetic data generator that works with local images
"""
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import os
import json
import glob
import sys
sys.path.append('..')  # Add parent directory to path
from loader import get_flickr_data

# ------------------------
# 🔧 Configurable Settings
# ------------------------
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
images_dir = "data/images"  # Local images directory
output_file = "syntetic_data/captions.json"
max_images = 10  # Limit for testing

# ------------------------
# 🚀 Load model & processor
# ------------------------
print("🔄 Loading model...")
try:
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Check for available devices in order of preference
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("🖥️  Using device: CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("🖥️  Using device: MPS (Apple GPU)")
else:
    device = torch.device('cpu')
    print("🖥️  Using device: CPU")

model = model.to(device)
# Only use half precision on CUDA, not on MPS (can cause issues)
if device.type == 'cuda':
    model = model.half()

# ------------------------
# 📦 Load Flickr8k images
# ------------------------
print("📂 Loading Flickr8k dataset...")
try:
    # Get the data loaders (this will download if needed)
    train_loader_fn, val_loader_fn = get_flickr_data(max_samples=100000, batch_size=1)
    
    # Collect all images from both train and val sets
    print("🔄 Collecting images from dataset...")
    all_images = []
    
    # Get training images
    for pil_images, captions in train_loader_fn():
        for img, caption in zip(pil_images, captions):
            all_images.append((img, caption))
    
    # Get validation images
    for pil_images, captions in val_loader_fn():
        for img, caption in zip(pil_images, captions):
            all_images.append((img, caption))
    
    # Limit for testing if needed
    if max_images > 0:
        all_images = all_images[:max_images]
    
    print(f"📷 Found {len(all_images)} images to process")
    
except Exception as e:
    print(f"❌ Error loading Flickr8k dataset: {e}")
    print("💡 Make sure you have internet connection for initial download")
    exit(1)

# ------------------------
# 📝 Define prompt - Fixed format for Qwen2.5-VL
# ------------------------
prompt = """<|user|>
<|image_start|><image><|image_end|>
Give me two captions for this image:

1. 😇 Bes (very supportive, encouraging):
2. 😈 Anti-Bes (extremely sarcastic, roasty, discouraging):

Be short, expressive, and funny when needed.
<|assistant|>"""

results = []

# Load existing results if the file exists
if os.path.exists(output_file):
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            # Get list of already processed images by index (since we don't have filenames)
            processed_indices = {result.get('image_index', -1) for result in existing_results}
            results = existing_results
            print(f"📂 Found existing results: {len(results)} captions already generated")
            print(f"🔄 Will skip {len(processed_indices)} already processed images")
    except Exception as e:
        print(f"⚠️ Could not load existing results: {e}")
        print("🔄 Starting fresh...")
        processed_indices = set()
else:
    processed_indices = set()

def save_results_incremental():
    """Save results to file with backup"""
    try:
        # Create backup if file exists
        if os.path.exists(output_file):
            backup_file = output_file.replace('.json', '_backup.json')
            os.rename(output_file, backup_file)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"💾 Saved {len(results)} captions to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

# ------------------------
# 🔁 Process images
# ------------------------
for i, (image, original_caption) in enumerate(all_images):
    try:
        # Skip if already processed
        if i in processed_indices:
            print(f"⏭️  Skipping image {i} (already processed)")
            continue
            
        print(f"\n🔄 Processing image {i+1}/{len(all_images)}...")
        print(f"📝 Original caption: {original_caption[:60]}{'...' if len(original_caption) > 60 else ''}")
        print(f"✅ Image loaded: {image.size}")
        
        # Process with model - Fixed processor call
        try:
            # Simple approach that works with Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                                                {
                            "type": "text", 
                            "text": "Write exactly 2 captions for this image:\n\n1. Positive caption (overly optimistic, find beauty in anything):\n2. Negative caption (brutally honest, sarcastic roast):\n\nKeep each caption under 25 words. Be creative and funny."
                        }
                    ],
                }
            ]
            
            # Apply chat template and process - simplified approach
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            
            # Only use half precision on CUDA, not on MPS
            if device.type == 'cuda':
                for k in inputs:
                    if torch.is_floating_point(inputs[k]):
                        inputs[k] = inputs[k].half()
            
            print("🔄 Generating captions...")
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                generated_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            print("✅ Generation complete!")
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"⚠️ GPU OOM for image {i}, skipping...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                continue
            else:
                raise
        
        # Parse captions - improved parsing logic
        bes_caption = ""
        anti_bes_caption = ""
        lines = generated_text.split('\n')
        
        # First try to find the specific format we requested
        for line in lines:
            line_lower = line.lower()
            if 'bes (' in line_lower and ':' in line:
                bes_caption = line.split(':', 1)[1].strip()
            elif 'anti-bes (' in line_lower and ':' in line:
                anti_bes_caption = line.split(':', 1)[1].strip()
            elif 'bes:' in line_lower and not bes_caption:
                bes_caption = line.split(':', 1)[1].strip()
            elif 'anti-bes:' in line_lower:
                anti_bes_caption = line.split(':', 1)[1].strip()
        
        # If that didn't work, try to parse numbered captions
        if not bes_caption and not anti_bes_caption:
            print("🔄 Trying numbered caption parsing...")
            caption_lines = []
            
            # Look for numbered captions more flexibly
            for line in lines:
                line = line.strip()
                # Try different numbered formats
                if (line.startswith('1.') or line.startswith('2.') or 
                    line.startswith('1)') or line.startswith('2)') or
                    'positive caption' in line.lower() or 'negative caption' in line.lower()):
                    
                    # Extract caption after the number/label
                    if ':' in line:
                        caption = line.split(':', 1)[1].strip()
                    elif line.startswith(('1.', '2.', '1)', '2)')):
                        caption = line[2:].strip()
                    else:
                        continue
                    
                    # Clean up the caption
                    if caption.startswith('"') and caption.endswith('"'):
                        caption = caption[1:-1]
                    
                    # Skip if it's just repeating the prompt
                    if (len(caption) > 10 and 
                        'toxic positivity' not in caption.lower() and
                        'obliterate any hope' not in caption.lower() and
                        'hit like a brick' not in caption.lower()):
                        caption_lines.append(caption)
            
            if len(caption_lines) >= 2:
                bes_caption = caption_lines[0]
                anti_bes_caption = caption_lines[1]
                print(f"✅ Extracted numbered captions: {len(caption_lines)} found")
            elif len(caption_lines) == 1:
                bes_caption = caption_lines[0]
                print("⚠️ Only one caption found, using as Bes caption")
            else:
                # Last resort: try to extract any meaningful sentences
                print("🔄 Trying fallback parsing...")
                sentences = []
                for line in lines:
                    line = line.strip()
                    if (len(line) > 10 and 
                        not line.lower().startswith('this image') and
                        not line.lower().startswith('the image') and
                        '.' in line and
                        len(line) < 200):  # Reasonable caption length
                        sentences.append(line)
                
                if len(sentences) >= 2:
                    bes_caption = sentences[0]
                    anti_bes_caption = sentences[1]
                    print(f"✅ Extracted fallback captions: {len(sentences)} found")
                elif len(sentences) == 1:
                    bes_caption = sentences[0]
                    print("⚠️ Only one fallback caption found")
        
        # Store results
        if bes_caption:
            print(f"😇 Bes: {bes_caption}")
            results.append({
                "image_index": i,
                "original_caption": original_caption,
                "caption": bes_caption,
                "type": "Bes"
            })
        
        if anti_bes_caption:
            print(f"😈 Anti-Bes: {anti_bes_caption}")
            results.append({
                "image_index": i,
                "original_caption": original_caption,
                "caption": anti_bes_caption,
                "type": "Anti-Bes"
            })
        
        if not bes_caption and not anti_bes_caption:
            print("⚠️ No captions extracted")
            print(f"Raw output: {generated_text}")
            print("🔍 Debug - Lines found:")
            for i, line in enumerate(lines):
                if line.strip():
                    print(f"  {i}: {line.strip()}")
        else:
            print("✅ Successfully extracted captions")
        
        # Clean up memory
        del inputs, generated_ids
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        
        # Save every 50 images or if we have new results
        if (i + 1) % 50 == 0 or (len(results) > 0 and len(results) % 50 == 0):
            print(f"\n💾 Checkpoint: Saving results after {i+1} images processed...")
            save_results_incremental()
            
            # Show current progress
            bes_count = sum(1 for r in results if r['type'] == 'Bes')
            anti_bes_count = sum(1 for r in results if r['type'] == 'Anti-Bes')
            print(f"📊 Progress: {len(results)} total captions ({bes_count} Bes, {anti_bes_count} Anti-Bes)")
            print(f"🔄 Continuing with remaining images...")
            
    except Exception as e:
        print(f"❌ Error processing image {i}: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        continue

# ------------------------
# 💾 Final save and summary
# ------------------------
print(f"\n🏁 Processing complete! Final save...")
if save_results_incremental():
    print(f"✅ Final results saved to {output_file}")
    
    # Show final statistics
    total_processed = len([idx for idx in range(len(all_images)) if idx not in processed_indices or any(r.get('image_index', -1) == idx for r in results)])
    print(f"📊 Final Summary:")
    print(f"  • Total images found: {len(all_images)}")
    print(f"  • Images processed: {total_processed}")
    print(f"  • Total captions generated: {len(results)}")
    
    if len(results) > 0:
        # Show detailed statistics
        bes_count = sum(1 for r in results if r['type'] == 'Bes')
        anti_bes_count = sum(1 for r in results if r['type'] == 'Anti-Bes')
        print(f"  • Caption breakdown: {bes_count} Bes, {anti_bes_count} Anti-Bes")
        
        # Show a sample of recent captions
        print(f"\n🔍 Sample captions (last 4):")
        for result in results[-4:]:  # Show last 4
            print(f"  {result['type']}: {result['caption'][:60]}{'...' if len(result['caption']) > 60 else ''}")
    else:
        print("⚠️ No captions were generated. Check the model output format.")
else:
    print("❌ Final save failed - check the backup files") 