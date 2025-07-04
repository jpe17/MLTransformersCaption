import os, json, torch, random
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def get_dual_caption_data(max_samples=100000, val_split=0.2, batch_size=8):
    """
    Load data with both positive (Bes) and negative (Anti-Bes) captions.
    Returns data loaders that yield (images, pos_captions, neg_captions, caption_types)
    """
    
    # Load the captions data
    if not os.path.exists("data/captions.json"):
        raise FileNotFoundError("data/captions.json not found. Please ensure your dual caption data is available.")
    
    print("📂 Loading dual caption data...")
    with open("syntetic_data/captions_backup.json", 'r') as f:
        all_captions = json.load(f)
    
    # Group by image - each image should have both Bes and Anti-Bes captions
    image_data = {}
    for item in all_captions:
        image_file = item['original_caption']  # This contains the image filename
        if image_file not in image_data:
            image_data[image_file] = {'Bes': [], 'Anti-Bes': []}
        
        if item['type'] == 'Bes':
            image_data[image_file]['Bes'].append(item['caption'])
        elif item['type'] == 'Anti-Bes':
            image_data[image_file]['Anti-Bes'].append(item['caption'])
    
    # Filter to only include images that have both types and exist on disk
    valid_data = []
    for image_file, captions in image_data.items():
        if len(captions['Bes']) > 0 and len(captions['Anti-Bes']) > 0:
            if os.path.exists(f"data/images/{image_file}"):
                valid_data.append({
                    'image': image_file,
                    'bes_captions': captions['Bes'],
                    'anti_bes_captions': captions['Anti-Bes']
                })
    
    print(f"📊 Found {len(valid_data)} images with both positive and negative captions")
    
    # Limit samples and shuffle
    if len(valid_data) > max_samples:
        valid_data = valid_data[:max_samples]
    random.shuffle(valid_data)
    
    # Split into train and validation
    split = int(len(valid_data) * (1 - val_split))
    train_data, val_data = valid_data[:split], valid_data[split:]
    
    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Image preprocessing
    resize_transform = Resize((224, 224))
    
    def batches(data):
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Load images
            pil_images = []
            pos_captions = []
            neg_captions = []
            
            for item in batch:
                # Load image
                try:
                    img = Image.open(f"data/images/{item['image']}").convert('RGB')
                    pil_images.append(resize_transform(img))
                except:
                    pil_images.append(Image.new('RGB', (224, 224)))  # Fallback
                
                # Randomly select one caption of each type
                pos_captions.append(random.choice(item['bes_captions']))
                neg_captions.append(random.choice(item['anti_bes_captions']))
            
            yield pil_images, pos_captions, neg_captions
    
    return lambda: batches(train_data), lambda: batches(val_data)

def get_single_caption_data_for_inference(image_paths):
    """
    Simple loader for inference - just images without captions
    """
    resize_transform = Resize((224, 224))
    
    pil_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            pil_images.append(resize_transform(img))
        except:
            pil_images.append(Image.new('RGB', (224, 224)))
    
    return pil_images 