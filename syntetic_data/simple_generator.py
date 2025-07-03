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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

model = model.to(device)
if device.type == 'cuda':
    model = model.half()

# ------------------------
# 📦 Find local images
# ------------------------
# Find all .jpg files in the images directory
image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
image_files = []

for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(images_dir, ext)))

if not image_files:
    print(f"❌ No images found in {images_dir}")
    print("💡 Please make sure there are .jpg files in the data/images folder")
    exit(1)

# Limit for testing if needed
if max_images > 0:
    image_files = image_files[:max_images]

print(f"📷 Found {len(image_files)} images to process")

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

# ------------------------
# 🔁 Process images
# ------------------------
for i, image_path in enumerate(image_files):
    try:
        image_name = os.path.basename(image_path)
        print(f"\n🔄 Processing {image_name} ({i+1}/{len(image_files)})...")
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"✅ Image loaded: {image.size}")
        except Exception as e:
            print(f"❌ Error loading image {image_name}: {e}")
            continue
        
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
                print(f"⚠️ CUDA OOM for {image_name}, skipping...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
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
                "image": image_name,
                "caption": bes_caption,
                "type": "Bes"
            })
        
        if anti_bes_caption:
            print(f"😈 Anti-Bes: {anti_bes_caption}")
            results.append({
                "image": image_name,
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
            
    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        continue

# ------------------------
# 💾 Save results
# ------------------------
try:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Done! Saved {len(results)} captions to {output_file}")
    print(f"📊 Processed {len(image_files)} images, generated {len(results)} captions")
    
    if len(results) > 0:
        # Show some statistics
        bes_count = sum(1 for r in results if r['type'] == 'Bes')
        anti_bes_count = sum(1 for r in results if r['type'] == 'Anti-Bes')
        print(f"📈 Caption breakdown: {bes_count} Bes, {anti_bes_count} Anti-Bes")
        
        # Show a sample
        print("\n🔍 Sample captions:")
        for i, result in enumerate(results[:4]):  # Show first 4
            print(f"  {result['type']}: {result['caption'][:60]}{'...' if len(result['caption']) > 60 else ''}")
    else:
        print("⚠️ No captions were generated. Check the model output format.")
        
except Exception as e:
    print(f"❌ Error saving results: {e}")
    print("💡 Results were not saved to file") 