from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from PIL import Image
import torch
import os
import json
from io import BytesIO
import requests

# ------------------------
# 🔧 Configurable Settings
# --------------git ----------
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
dataset_id = "jpe17/CaptionGenerator1.0"
output_file = "syntetic_data/captions.json"
preview = True         # ✅ Set to False to run on full dataset
preview_count = 10     # Number of samples to process in preview mode

# ------------------------
# 🚀 Load model & processor
# ------------------------
if "local_model_dir" in os.environ:
    processor = AutoProcessor.from_pretrained("local_model_dir", use_fast=True)
    model = AutoModelForVision2Seq.from_pretrained("local_model_dir")
else:
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForVision2Seq.from_pretrained(model_id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if device.type == 'cuda':
    model = model.half()

# ------------------------
# 📦 Load dataset
# ------------------------
dataset = load_dataset(dataset_id, split="train")

# Optional preview
if preview:
    dataset = dataset.select(range(min(preview_count, len(dataset))))

# ------------------------
# 📝 Define prompt
# ------------------------
prompt = """<|user|>
<|image_start|><image><|image_end|>
Give me two captions for this image:

1. 😇 Bes (supportive, encouraging):
2. 😈 Anti-Bes (sarcastic, roasty, discouraging):

Be short, expressive, and funny when needed.
<|assistant|>"""

results = []

# ------------------------
# 🔁 Loop through dataset
# ------------------------
for i, example in enumerate(dataset):
    try:
        # Load image (assuming it's in 'image' field)
        image_data = example["image"]
        image_name = example.get("file_name", f"image_{i}.jpg")

        image = None # Initialize image as None

        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, torch.Tensor):
            # If image_data is already a tensor, convert it to PIL Image
            # Assuming C, H, W format, and values might be normalized (e.g., to [-1, 1] or [0, 1])
            # We'll denormalize to 0-255 and convert to uint8.
            # This is a common conversion, but may need adjustment if different normalization is used.
            image_tensor_cpu = image_data.cpu().detach().float() # Ensure float type

            # Normalize to [0, 1] range if not already, then scale to [0, 255]
            # Handle potential different input tensor value ranges.
            # A common range is [0, 1] or [-1, 1].
            # If the tensor has values outside [0, 1] and contains negative numbers,
            # it's likely normalized to [-1, 1] or similar.
            if image_tensor_cpu.min() < 0 or image_tensor_cpu.max() > 1:
                # If values are not in [0, 1] or [-1, 1], attempt min-max scaling to [0, 1]
                min_val = image_tensor_cpu.min()
                max_val = image_tensor_cpu.max()
                if max_val != min_val: # Avoid division by zero
                    image_tensor_normalized_0_1 = (image_tensor_cpu - min_val) / (max_val - min_val)
                else:
                    image_tensor_normalized_0_1 = torch.zeros_like(image_tensor_cpu) # All values are same
            else:
                image_tensor_normalized_0_1 = image_tensor_cpu # Already in [0, 1] or close enough

            # Scale to [0, 255] and convert to uint8
            image_tensor_255 = (image_tensor_normalized_0_1 * 255).clamp(0, 255).to(torch.uint8)

            # Permute from C, H, W to H, W, C for PIL.Image.fromarray
            image_np = image_tensor_255.permute(1, 2, 0).numpy()
            image = Image.fromarray(image_np, 'RGB')

        elif hasattr(image_data, "convert"):
            # This covers PIL.Image objects directly from dataset
            image = image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(BytesIO(image_data["bytes"])) .convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

        if image is None:
            raise ValueError("Image could not be loaded or converted to PIL Image.")

        print(f"Processing {image_name}...")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        if device.type == 'cuda':
            for k in inputs:
                if torch.is_floating_point(inputs[k]):
                    inputs[k] = inputs[k].half()

        print(f"Generating for {image_name}...")
        try:
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=128)
                generated_ids = output[0][len(inputs.input_ids[0]):]
                generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"⚠️ CUDA OOM for {image_name}, skipping and clearing cache.")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                raise
        print(f"Done generating for {image_name}")

        # Parse captions
        bes_caption = ""
        anti_bes_caption = ""
        lines = generated_text.split('\n')
        for line in lines:
            if 'bes:' in line.lower():
                bes_caption = line.split(':', 1)[1].strip()
            elif 'anti-bes:' in line.lower():
                anti_bes_caption = line.split(':', 1)[1].strip()

        print(f"\n📷 {image_name}")
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
        print()

        # Explicitly delete tensors and clear cache to free memory
        del inputs
        if 'output' in locals():
            del output
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️ Error processing image {i}: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        continue

# ------------------------
# 💾 Save results
# ------------------------
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Done! Saved {len(results)} captions to {output_file}")
