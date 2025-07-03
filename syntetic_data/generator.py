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
# ------------------------
model_id = "Qwen/Qwen1.5-VL-Chat"
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

        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        elif hasattr(image_data, "convert"):
            image = image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
        else:
            raise ValueError("Unsupported image format")

        print(f"Processing {image_name}...")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        print(f"Generating for {image_name}...")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=128)
            generated_ids = output[0][len(inputs.input_ids[0]):]
            generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
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

    except Exception as e:
        print(f"⚠️ Error processing image {i}: {e}")
        continue

# ------------------------
# 💾 Save results
# ------------------------
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Done! Saved {len(results)} captions to {output_file}")
