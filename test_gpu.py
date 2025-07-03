#!/usr/bin/env python3
"""
Simple GPU test to check if MPS/CUDA is working
"""
import torch
import time

def test_gpu():
    print("🔍 GPU Availability Test")
    print("=" * 40)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check available devices
    print("\n📱 Available devices:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
    
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  MPS built: {torch.backends.mps.is_built()}")
    
    # Determine best device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = "CUDA GPU"
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "MPS (Apple GPU)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    print(f"\n🖥️  Using device: {device_name}")
    
    # Performance test
    print("\n⚡ Performance Test:")
    print("  Creating tensors and performing operations...")
    
    # Test matrix multiplication
    size = 1000
    
    # CPU test
    start_time = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"  CPU time: {cpu_time:.4f} seconds")
    
    # GPU test (if available)
    if device.type != 'cpu':
        try:
            start_time = time.time()
            a_gpu = torch.randn(size, size, device=device)
            b_gpu = torch.randn(size, size, device=device)
            c_gpu = torch.mm(a_gpu, b_gpu)
            # Synchronize to ensure operation is complete
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            gpu_time = time.time() - start_time
            print(f"  {device_name} time: {gpu_time:.4f} seconds")
            print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
            
            # Memory test
            print(f"\n💾 Memory Test:")
            if device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
                print(f"  CUDA memory allocated: {memory_allocated:.2f} MB")
                print(f"  CUDA memory reserved: {memory_reserved:.2f} MB")
            elif device.type == 'mps':
                # MPS doesn't have memory stats like CUDA
                print(f"  MPS memory management is handled automatically")
            
        except Exception as e:
            print(f"  ❌ GPU test failed: {e}")
    
    print("\n✅ Test complete!")
    return device, device_name

if __name__ == "__main__":
    test_gpu() 