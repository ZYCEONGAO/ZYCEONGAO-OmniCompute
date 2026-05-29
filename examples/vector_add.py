import torch
import time
import sys

def main():
    print("========================================")
    print(" OmniCompute Zero-Invasion Demo")
    print("========================================")
    
    if not torch.cuda.is_available():
        print("[!] PyTorch does not detect CUDA. Make sure you run this script")
        print("    through the OmniCompute CLI wrapper:")
        print("    $ omnicompute run python vector_add.py")
        sys.exit(1)

    print(f"[*] Detected CUDA Devices: {torch.cuda.device_count()}")
    print(f"[*] Device Name: {torch.cuda.get_device_name(0)}")

    size = 100000000
    print(f"\n[*] Allocating tensors of size {size}...")
    
    start = time.time()
    
    # These operations will be transparently intercepted by omni-shim
    # and executed on AMD/Apple/Intel hardware via omni-core JIT
    a = torch.ones(size, device="cuda")
    b = torch.ones(size, device="cuda")
    
    # JIT trigger point
    c = a + b
    
    # Device to Host sync
    result = c[0].item()
    
    end = time.time()
    
    print(f"\n[*] Computation complete!")
    print(f"[*] Result: c[0] = {result} (Expected: 2.0)")
    print(f"[*] Time taken: {end - start:.4f} seconds")

if __name__ == "__main__":
    main()
