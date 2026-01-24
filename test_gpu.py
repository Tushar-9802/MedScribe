import torch
import transformers

print("="*60)
print("GPU Compatibility Test - RTX 5070 Ti")
print("="*60)

# PyTorch version
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test tensor operations
    print("\nTesting GPU tensor operations...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print("✓ Matrix multiplication works")
    
    # Test FP16
    x_fp16 = x.half()
    y_fp16 = y.half()
    z_fp16 = torch.matmul(x_fp16, y_fp16)
    print("✓ FP16 operations work")
    
    # Test bfloat16
    x_bf16 = x.bfloat16()
    y_bf16 = y.bfloat16()
    z_bf16 = torch.matmul(x_bf16, y_bf16)
    print("✓ BFloat16 operations work")
    
else:
    print("✗ CUDA not available - check installation")

print(f"\nTransformers: {transformers.__version__}")
print("\n✓✓✓ GPU test complete")