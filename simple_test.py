from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model(model_name, test_prompts):
    print(f"\n🧪 Testing: {model_name}")
    print("="*50)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        
        for prompt in test_prompts[:3]:  # Test first 3 prompts
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=15,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Get only new tokens
                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                print(f"'{prompt}' → '{generated.strip()}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    test_prompts = [
        "A beautiful",
        "The cat is", 
        "In the garden",
        "This picture shows"
    ]
    
    # Test different cached models
    models_to_test = [
        "Qwen/Qwen2-0.5B-Instruct",  # Your cached instruct model
        "Qwen/Qwen3-0.6B-Base",      # Current base model
        "Qwen/Qwen2.5-1.5B-Instruct" # Larger instruct model
    ]
    
    for model_name in models_to_test:
        test_model(model_name, test_prompts)
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main() 