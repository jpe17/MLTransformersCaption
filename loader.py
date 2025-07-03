import os, json, torch, urllib.request as url, zipfile, random
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def get_flickr_data(max_samples=100000, val_split=0.2, batch_size=16):
    # Download images (Flickr8k)
    if not os.path.exists("data/images"):
        print("📸 Downloading Flickr8k images (1GB)...")
        os.makedirs("data", exist_ok=True)
        url.urlretrieve("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", "data/images.zip")
        with zipfile.ZipFile("data/images.zip") as z:
            z.extractall("data") # Extracts to data/Flicker8k_Dataset
        os.rename("data/Flicker8k_Dataset", "data/images")
        os.remove("data/images.zip")

    # Download and process captions (Flickr8k)
    if not os.path.exists("data/captions.json"):
        print("📂 Downloading Flickr8k captions...")
        url.urlretrieve("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", "data/captions.zip")
        with zipfile.ZipFile("data/captions.zip") as z:
            z.extractall("data")
        
        captions_file = "data/Flickr8k.token.txt"
        captions_data = []
        with open(captions_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_id, caption = parts[0][:-2], parts[1] # "image.jpg#0" -> "image.jpg"
                    captions_data.append({'image': image_id, 'caption': caption})
        
        json.dump(captions_data, open("data/captions.json", "w"))
        # Clean up
        os.remove("data/captions.zip")
        os.remove(captions_file)
        for f in ["Flickr_8k.trainImages.txt", "Flickr_8k.devImages.txt", "Flickr_8k.testImages.txt", "readme.txt"]:
            if os.path.exists(f"data/{f}"): os.remove(f"data/{f}")

    # Load and split
    data = json.load(open("data/captions.json"))[:max_samples]
    random.shuffle(data)
    split = int(len(data) * (1 - val_split))
    train_data, val_data = data[:split], data[split:]
    
    # Just resize PIL images - let each model handle its own preprocessing
    resize_transform = Resize((224, 224))
    
    # Batch generators
    def batches(data_list, shuffle=False):
        if shuffle:
            random.shuffle(data_list)
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            # Load and resize PIL images once
            pil_images = []
            for item in batch:
                if os.path.exists(f"data/images/{item['image']}"):
                    img = Image.open(f"data/images/{item['image']}").convert('RGB')
                    pil_images.append(resize_transform(img))
                else:
                    pil_images.append(Image.new('RGB', (224, 224)))  # Empty image fallback
            
            captions = [item['caption'] for item in batch]
            yield pil_images, captions
    
    return lambda: batches(train_data, shuffle=True), lambda: batches(val_data)
