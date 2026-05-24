import torch
import numpy as np
import sys
import os

# Add workspace to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.components.encoder import RepDSConv

def test_reparam_equivalence():
    print("Running RepDSConv Equivalence Unit Test...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    in_channels = 32
    out_channels = 64
    height, width = 16, 16
    
    # Initialize RepDSConv in training mode
    block = RepDSConv(in_channels, out_channels, deploy=False)
    
    # Generate random input tensor
    x = torch.randn(2, in_channels, height, width)
    
    # Set to eval state (so batch norm uses running stats instead of updating them)
    block.eval()
    
    # Forward pass in training (eval) mode
    with torch.no_grad():
        out_train = block(x)
    
    # Switch block to deploy mode (reparameterize and fuse BN layers)
    block.switch_to_deploy()
    
    # Forward pass in deploy mode
    with torch.no_grad():
        out_deploy = block(x)
    
    # Compare outputs
    difference = torch.abs(out_train - out_deploy)
    max_diff = torch.max(difference).item()
    mean_diff = torch.mean(difference).item()
    
    print(f"-> Max absolute difference: {max_diff:.8f}")
    print(f"-> Mean absolute difference: {mean_diff:.8f}")
    
    # Verify they match closely (typical tolerance for float32 precision is 1e-5)
    assert max_diff < 1e-5, f"Reparameterization output mismatch! Max diff: {max_diff}"
    print("[PASS] RepDSConv Equivalence Test PASSED successfully!\n")

if __name__ == "__main__":
    test_reparam_equivalence()
