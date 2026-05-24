import torch
import sys
import os

# Add workspace to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.components.decoder import DecoderBlock

def test_decoder_block():
    print("Running DecoderBlock Unit Test...")
    torch.manual_seed(42)
    
    in_channels = 128
    skip_channels = 64
    out_channels = 32
    
    # 1. Test in training mode (deploy=False)
    block = DecoderBlock(in_channels, skip_channels, out_channels, deploy=False)
    
    # Feature maps
    x = torch.randn(2, in_channels, 16, 32)
    skip = torch.randn(2, skip_channels, 32, 64) # Encoder feature map has 2x spatial size
    
    out = block(x, skip)
    print(f"-> Input x shape: {x.shape}")
    print(f"-> Input skip shape: {skip.shape}")
    print(f"-> Output shape: {out.shape}")
    
    # Check spatial dimensions (should be upsampled to match skip features: 32x64)
    assert out.shape == (2, out_channels, 32, 64), f"Incorrect output shape! Expected (2, 32, 32, 64), got {out.shape}"
    
    # 2. Test in deploy mode
    block.eval()
    block.switch_to_deploy()
    
    out_deploy = block(x, skip)
    assert out_deploy.shape == (2, out_channels, 32, 64), f"Incorrect output shape in deploy mode! Expected (2, 32, 32, 64), got {out_deploy.shape}"
    
    print("[PASS] DecoderBlock Unit Test PASSED successfully!\n")

if __name__ == "__main__":
    test_decoder_block()
