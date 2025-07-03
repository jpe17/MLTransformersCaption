from model import TextOnlyDecoder
import torch

def test_text_decoder_interactive():
    """Interactive testing of the text-only decoder"""
    print("🔤 Text-Only Decoder Test")
    print("=" * 50)
    
    # Initialize the decoder
    print("Loading text-only decoder...")
    decoder = TextOnlyDecoder()
    decoder.eval()
    print("✅ Decoder loaded successfully!")
    print()
    
    # Predefined test prompts
    test_prompts = [
        "A beautiful",
        "The cat is",
        "In the garden",
        "This picture shows",
        "A person walking",
        "The weather is",
        "I can see",
        "The sky looks",
        "A dog running",
        "The mountain is"
    ]
    
    print("🧪 Testing predefined prompts:")
    print("-" * 30)
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            generated = decoder(prompt, max_length=15)
            print(f"{i:2d}. Input:  '{prompt}'")
            print(f"    Output: '{prompt} {generated}'")
            print()
    
    print("=" * 50)
    print("💡 Interactive mode - Enter your own prompts!")
    print("   (Type 'quit' to exit)")
    print()
    
    # Interactive mode
    while True:
        try:
            user_input = input("Enter a prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate with different lengths
            print(f"\n📝 Generating from: '{user_input}'")
            print("-" * 30)
            
            with torch.no_grad():
                for length in [10, 15, 20]:
                    generated = decoder(user_input, max_length=length)
                    print(f"Length {length:2d}: '{user_input} {generated}'")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

def test_decoder_comparison():
    """Compare different generation parameters"""
    print("🔬 Decoder Parameter Comparison")
    print("=" * 50)
    
    decoder = TextOnlyDecoder()
    decoder.eval()
    
    test_prompt = "A beautiful landscape"
    
    print(f"Testing with prompt: '{test_prompt}'")
    print("=" * 50)
    
    with torch.no_grad():
        for max_len in [5, 10, 15, 20, 25]:
            generated = decoder(test_prompt, max_length=max_len)
            print(f"Max length {max_len:2d}: '{test_prompt} {generated}'")
    
    print("\n" + "=" * 50)
    print("🎯 Multiple generations from same prompt (randomness test):")
    print("=" * 50)
    
    # Generate multiple times to see consistency
    for i in range(5):
        with torch.no_grad():
            generated = decoder(test_prompt, max_length=15)
            print(f"Generation {i+1}: '{test_prompt} {generated}'")

if __name__ == "__main__":
    print("Select test mode:")
    print("1. Interactive testing")
    print("2. Parameter comparison")
    print("3. Both")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_text_decoder_interactive()
    elif choice == "2":
        test_decoder_comparison()
    elif choice == "3":
        test_decoder_comparison()
        print("\n" + "=" * 70)
        print("\n")
        test_text_decoder_interactive()
    else:
        print("Invalid choice. Running interactive mode...")
        test_text_decoder_interactive() 