"""
Test script to verify GPU/CUDA availability
"""
import torch
import sys

print("="*60)
print("PyTorch GPU Detection Test")
print("="*60)

# PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test tensor creation on GPU
    try:
        test_tensor = torch.randn(100, 100).cuda()
        print(f"\n✓ Successfully created tensor on GPU: {test_tensor.device}")
    except Exception as e:
        print(f"\n✗ Error creating tensor on GPU: {e}")
else:
    print("\n⚠ CUDA is not available. Checking reasons:")
    print("  1. PyTorch may not be installed with CUDA support")
    print("  2. NVIDIA drivers may not be installed")
    print("  3. CUDA toolkit may not be installed")
    print("\nTo install PyTorch with CUDA support, visit:")
    print("  https://pytorch.org/get-started/locally/")

# Check for MPS (Apple Silicon)
if hasattr(torch.backends, 'mps'):
    print(f"\nMPS (Apple Metal) available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.randn(100, 100).to('mps')
            print(f"✓ Successfully created tensor on MPS: {test_tensor.device}")
        except Exception as e:
            print(f"✗ Error creating tensor on MPS: {e}")

# Recommended device
if torch.cuda.is_available():
    recommended = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    recommended = 'mps'
else:
    recommended = 'cpu'

print(f"\n{'='*60}")
print(f"Recommended device: {recommended}")
print(f"{'='*60}")
