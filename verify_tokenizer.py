from transformers import AutoTokenizer

def verify_qwen_tokenizer(model_name="Qwen/Qwen3-0.6B-Base"):
    """
    Loads a tokenizer and prints its special token configuration to verify
    its properties, especially the presence or absence of a BOS token.
    """
    print("="*50)
    print(f"🔎 Verifying tokenizer for: {model_name}")
    print("="*50)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"  -> BOS Token: {tokenizer.bos_token}")
        print(f"  -> BOS Token ID: {tokenizer.bos_token_id}")
        print("-" * 25)
        print(f"  -> EOS Token: {tokenizer.eos_token}")
        print(f"  -> EOS Token ID: {tokenizer.eos_token_id}")
        print("-" * 25)
        print(f"  -> PAD Token: {tokenizer.pad_token}")
        print(f"  -> PAD Token ID: {tokenizer.pad_token_id}")
        
        print("\n✅ Verification complete.")
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")

if __name__ == "__main__":
    verify_qwen_tokenizer() 