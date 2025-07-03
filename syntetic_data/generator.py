from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import os
import json

# Load model + processor
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = AutoModelForVision2Seq.from_pretrained("local_model_dir")
processor = AutoProcessor.from_pretrained("local_model_dir")

# Setup paths
image_dir = "syntetic_data/images"
output_file = "syntetic_data/captions.json"
results = []

# Format prompt
prompt = """<|user|>
<|image_start|><image><|image_end|>
Give me two captions for this image:

1. 😇 Bes (supportive, encouraging):
2. 😈 Anti-Bes (sarcastic, roasty, discouraging):

Be short, expressive, and funny when needed.
<|assistant|>"""

# Get list of images
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"Processing {image_path}...")
    
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
        generated_ids = output[0][len(inputs.input_ids[0]):]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"Generated text: {generated_text}")

        # Parse generated text
        bes_caption = ""
        anti_bes_caption = ""
        try:
            lines = generated_text.split('\n')
            for line in lines:
                if 'bes:' in line.lower():
                    bes_caption = line.split(':', 1)[1].strip()
                elif 'anti-bes:' in line.lower():
                    anti_bes_caption = line.split(':', 1)[1].strip()
        except Exception as e:
            print(f"Error parsing output for {image_file}: {e}")

        if bes_caption:
            results.append({
                "image": image_file,
                "caption": bes_caption
            })
        
        if anti_bes_caption:
            results.append({
                "image": image_file,
                "caption": anti_bes_caption
            })

# Save results
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Done. Results saved to {output_file}")


