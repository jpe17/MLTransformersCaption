import torch
from transformers import get_linear_schedule_with_warmup
import itertools
import wandb
import os
import random
import numpy as np
from PIL import Image
from model_explained import VisionLanguageEncoder as VisionLanguageEncoderBase, CaptionDecoder
from loader import get_flickr_data

# --- Sweep Parameter Space ---
SWEEP_SPACE = {
    'num_epochs': 1,
    'grad_clip_norm': (0.8, 1.2),
    'learning_rate': (1e-5, 5e-5),
    'weight_decay': (0.005, 0.02),
    'beta1': 0.9,
    'beta2': (0.995, 0.999),
    'top_k': (40, 60),
    'temperature': (0.9, 1.1),
}

# --- Model Definition ---
class VisionLanguageEncoderWithModality(VisionLanguageEncoderBase):
    """
    Version with modality embeddings, ready for training.
    """
    def forward(self, pil_images, captions):
        device = next(self.qwen_model.parameters()).device
        
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt")
            clip_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in clip_inputs.items()}
            vision_outputs = self.clip_model.vision_model(**clip_inputs)
            image_patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]

        image_patch_embeddings = self.image_adapter(image_patch_embeddings)

        tokenized = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        tokens = tokenized['input_ids'].to(device)
        
        sos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        eos = self.tokenizer.eos_token_id
        
        sos_tokens = torch.full((tokens.shape[0], 1), sos, dtype=tokens.dtype, device=device)
        input_padded = torch.cat([sos_tokens, tokens], dim=1)
        
        eos_tokens = torch.full((tokens.shape[0], 1), eos, dtype=tokens.dtype, device=device)
        target_padded = torch.cat([tokens, eos_tokens], dim=1)

        with torch.no_grad():
            text_embeddings = self.qwen_model.get_input_embeddings()(input_padded)

        image_ids = torch.zeros(image_patch_embeddings.shape[:2], dtype=torch.long, device=device)
        image_mod_embs = self.modality_embedding(image_ids)
        final_image_embeddings = image_patch_embeddings + image_mod_embs

        text_ids = torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=device)
        text_mod_embs = self.modality_embedding(text_ids)
        final_text_embeddings = text_embeddings + text_mod_embs

        return final_image_embeddings, final_text_embeddings, target_padded

def sample_config(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    config = {}
    config['num_epochs'] = SWEEP_SPACE['num_epochs']
    config['grad_clip_norm'] = random.uniform(*SWEEP_SPACE['grad_clip_norm'])
    config['learning_rate'] = 10 ** random.uniform(np.log10(SWEEP_SPACE['learning_rate'][0]), np.log10(SWEEP_SPACE['learning_rate'][1]))
    config['weight_decay'] = 10 ** random.uniform(np.log10(SWEEP_SPACE['weight_decay'][0]), np.log10(SWEEP_SPACE['weight_decay'][1]))
    config['beta1'] = SWEEP_SPACE['beta1']
    config['beta2'] = random.uniform(*SWEEP_SPACE['beta2'])
    config['top_k'] = random.randint(*SWEEP_SPACE['top_k'])
    config['temperature'] = random.uniform(*SWEEP_SPACE['temperature'])
    config['seed'] = seed
    return config

def train_sweepable(config):
    wandb.init(project="image-captioning-v2-sweep", config=config, reinit=True)
    config = wandb.config
    seed = config.get('seed', None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Config: {dict(config)}")
    print("Initializing model and data...")
    train_loader_fn, val_loader_fn = get_flickr_data() 
    encoder = VisionLanguageEncoderWithModality().to(device)
    decoder = CaptionDecoder().to(device)
    trainable_params = list(encoder.image_adapter.parameters()) + \
                       list(encoder.modality_embedding.parameters()) + \
                       list(decoder.vision_cross_attention.parameters()) + \
                       list(decoder.attn_layer_norm.parameters()) + \
                       [p for p in decoder.qwen_model.model.layers[-2:].parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    num_epochs = config.num_epochs
    print(f"Starting training for {num_epochs} epochs...")
    total_loss = 0
    step_count = 0
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_batches = train_loader_fn()
        encoder.train()
        decoder.train()
        for step, (pil_images, captions) in enumerate(train_batches):
            image_embeddings, text_embeddings, target_tokens = encoder(pil_images, captions)
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                _, loss = decoder(image_embeddings, text_embeddings, target_tokens)
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at step {step}. Skipping.")
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            step_count += 1
            if step % 20 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")
                wandb.log({
                    "train_loss_step": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                })

    print("\n✅ Training complete! Running validation...")
    encoder.eval()
    decoder.eval()

    # --- Quantitative Validation (Loss) ---
    print("Calculating validation loss...")
    val_loss = 0
    val_steps = 0
    val_batches_for_loss = val_loader_fn()
    with torch.no_grad():
        for step, (pil_images, captions) in enumerate(itertools.islice(val_batches_for_loss, 20)):
            image_embeddings, text_embeddings, target_tokens = encoder(pil_images, captions)
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                _, loss = decoder(image_embeddings, text_embeddings, target_tokens)
            if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()
                val_steps += 1
    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    print(f"  Average validation loss over {val_steps} batches: {avg_val_loss:.4f}")

    # --- Qualitative Validation (Image Captioning Examples) ---
    print("Generating validation examples...")
    val_batches = val_loader_fn()
    validation_table = wandb.Table(columns=["Image", "True Caption", "Predicted Caption"])
    with torch.no_grad():
        for i, (pil_images, true_captions) in enumerate(itertools.islice(val_batches, 5)):
            first_image_pil = pil_images[0]
            image_embeddings, _, _ = encoder([first_image_pil], ["dummy caption"])
            generated_ids = []
            sos_id = decoder.tokenizer.bos_token_id or decoder.tokenizer.eos_token_id
            input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)
            for _ in range(30):
                text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
                text_mod_id = torch.ones_like(input_ids, device=device)
                text_mod_emb = encoder.modality_embedding(text_mod_id)
                text_embeddings_final = text_embeddings + text_mod_emb
                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                    logits, _ = decoder(image_embeddings, text_embeddings_final)
                next_token_logits = logits[:, -1, :]
                k = config.top_k
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                if config.temperature > 0:
                    top_k_logits /= config.temperature
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token_id = top_k_indices.gather(-1, next_token_idx)
                if next_token_id.item() == decoder.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            true_caption = true_captions[0]
            print(f"\nImage {i+1}:")
            print(f"  True: {true_caption}")
            print(f"  Pred: {predicted_caption}")
            validation_table.add_data(wandb.Image(first_image_pil), true_caption, predicted_caption)
    avg_train_loss = total_loss / step_count if step_count > 0 else 0
    wandb.log({
        "final_avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "validation_examples": validation_table
    })
    
    print(f"\nFinal Average Training Loss: {avg_train_loss:.4f}")

    # --- Inference on a specific image ---
    print("\n--- Running inference on test_01.png ---")
    try:
        image_path = "test_01.png"
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping inference on this image.")
        else:
            pil_image = Image.open(image_path).convert("RGB")
            
            image_embeddings, _, _ = encoder([pil_image], ["dummy caption"])
            generated_ids = []
            sos_id = decoder.tokenizer.bos_token_id or decoder.tokenizer.eos_token_id
            input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)
            for _ in range(30):
                text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
                text_mod_id = torch.ones_like(input_ids, device=device)
                text_mod_emb = encoder.modality_embedding(text_mod_id)
                text_embeddings_final = text_embeddings + text_mod_emb
                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                    logits, _ = decoder(image_embeddings, text_embeddings_final)
                next_token_logits = logits[:, -1, :]
                k = config.top_k
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                if config.temperature > 0:
                    top_k_logits /= config.temperature
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token_id = top_k_indices.gather(-1, next_token_idx)
                if next_token_id.item() == decoder.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nImage: {image_path}")
            print(f"  Predicted Caption: {predicted_caption}")
            
            wandb.log({
                "test_inference_image": wandb.Image(pil_image, caption=f"Predicted: {predicted_caption}")
            })

    except Exception as e:
        print(f"An error occurred during inference on {image_path}: {e}")

    print("Sweep run finished!")

if __name__ == "__main__":
    for sweep_id in range(10):
        config = sample_config(seed=42 + sweep_id)
        print(f"\n========== SWEEP RUN {sweep_id+1}/10 ==========")
        train_sweepable(config) 