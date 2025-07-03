#!/usr/bin/env python3
"""
Quick Start Script for Synthetic Data Generation
"""
import os
import sys
import subprocess

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("🚀 Synthetic Data Generator - Quick Start")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Warning: Not in a virtual environment")
        print("💡 Consider running: python -m venv venv && source venv/bin/activate")
        print()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import torch
        import transformers
        import PIL
        import datasets
        print("✅ All dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install -r requirements.txt")
        return
    
    print()
    print("Choose an option:")
    print("1. 🧪 Test with a simple generated image (recommended first)")
    print("2. 🖼️  Use local images from data/images folder")
    print("3. 🌐 Use HuggingFace dataset (original script)")
    print("4. 📋 Show requirements.txt")
    print("5. 🔧 Install dependencies")
    print()
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🧪 Running test generator...")
        success, output = run_command("python syntetic_data/test_generator.py")
        if success:
            print("✅ Test completed successfully!")
            print("📁 Check syntetic_data/test_captions.json for results")
        else:
            print("❌ Test failed:")
            print(output)
    
    elif choice == "2":
        print("\n🖼️  Running local image generator...")
        image_count = len([f for f in os.listdir("data/images") if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
        print(f"📷 Found {image_count} images in data/images")
        
        if image_count == 0:
            print("❌ No images found in data/images folder")
            print("💡 Please add some images to data/images/")
            return
        
        success, output = run_command("python syntetic_data/simple_generator.py")
        if success:
            print("✅ Generation completed!")
            print("📁 Check syntetic_data/captions.json for results")
        else:
            print("❌ Generation failed:")
            print(output)
    
    elif choice == "3":
        print("\n🌐 Running HuggingFace dataset generator...")
        print("⚠️  This will download data from HuggingFace")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            success, output = run_command("python syntetic_data/generator.py")
            if success:
                print("✅ Generation completed!")
                print("📁 Check syntetic_data/captions.json for results")
            else:
                print("❌ Generation failed:")
                print(output)
        else:
            print("❌ Cancelled")
    
    elif choice == "4":
        print("\n📋 Requirements:")
        try:
            with open("requirements.txt", "r") as f:
                print(f.read())
        except FileNotFoundError:
            print("❌ requirements.txt not found")
    
    elif choice == "5":
        print("\n🔧 Installing dependencies...")
        success, output = run_command("pip install -r requirements.txt")
        if success:
            print("✅ Dependencies installed successfully!")
        else:
            print("❌ Installation failed:")
            print(output)
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 