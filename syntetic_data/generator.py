from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import os
import json

# === Settings ===
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
image_path = "syntetic_data/test_image.jpg"
output_file = "syntetic_data/captions.json"

# === Load model & processor ===
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    model = model.half()

# === Load image ===
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
image = Image.open(image_path).convert("RGB")

# === Define prompt ===
prompt = """Give me two captions for this image:

1. 😇 Bes (very supportive, encouraging):
2. 😈 Anti-Bes (extremely sarcastic, roasty, discouraging):

Be short, expressive, and funny when needed."""

# === Format input for Qwen ===
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True,
).to(device)

if device.type == "cuda":
    for k in inputs:
        if torch.is_floating_point(inputs[k]):
            inputs[k] = inputs[k].half()

# === Generate output ===
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
    result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

# === Save result ===
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump({"image": os.path.basename(image_path), "output": result}, f, indent=2)

print("✅ Done!")
print(result)
