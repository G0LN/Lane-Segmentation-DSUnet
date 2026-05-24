import torch
import torch.nn as nn
from .components.encoder import EncoderBlock
from .components.decoder import DecoderBlock

class DSUnet(nn.Module):
    """
    DSUnet (Dual Stream Unet) Architecture for Lane Detection / Segmentation.
    Optimized with Structural Reparameterization, Asymmetric Decoder, and Skip Compression.
    """
    def __init__(self, in_channels=3, num_classes=4, dropout=0.5, width_multiplier=1.0, deploy=False):
        super(DSUnet, self).__init__()
        self.deploy = deploy
        
        # Scale number of channels dynamically based on width_multiplier (alpha)
        c1 = int(64 * width_multiplier)
        c2 = int(128 * width_multiplier)
        c3 = int(256 * width_multiplier)
        c4 = int(512 * width_multiplier)
        c5 = int(1024 * width_multiplier) # Bottleneck
        
        # Encoder Path
        self.enc1 = EncoderBlock(in_channels, c1, use_pool=True, deploy=deploy)
        self.enc2 = EncoderBlock(c1, c2, use_pool=True, deploy=deploy)
        self.enc3 = EncoderBlock(c2, c3, use_pool=True, deploy=deploy)
        self.enc4 = EncoderBlock(c3, c4, use_pool=True, dropout_prob=dropout, deploy=deploy)
        
        # Bottleneck (No pooling)
        self.bottleneck = EncoderBlock(c4, c5, use_pool=False, dropout_prob=dropout, deploy=deploy)
        
        # Decoder Path (Asymmetric: Bilinear Upsample + skip compression + single RepDSConv)
        self.dec4 = DecoderBlock(c5, c4, c4, dropout_prob=dropout, deploy=deploy)
        self.dec3 = DecoderBlock(c4, c3, c3, deploy=deploy)
        self.dec2 = DecoderBlock(c3, c2, c2, deploy=deploy)
        self.dec1 = DecoderBlock(c2, c1, c1, deploy=deploy)
        
        # Final prediction layer: 1x1 Standard Conv
        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoding
        skip1, p1 = self.enc1(x)
        skip2, p2 = self.enc2(p1)
        skip3, p3 = self.enc3(p2)
        skip4, p4 = self.enc4(p3)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoding
        d4 = self.dec4(b, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        
        out = self.out_conv(d1)
        return out

    def switch_to_deploy(self):
        """
        Recursively converts all Reparameterizable convolutions (RepDSConv)
        in the model into deploy mode (fuses parallel branches into single convs).
        """
        if self.deploy:
            return
            
        print("Fusing model branches (switch_to_deploy)...")
        for m in self.modules():
            if m is not self and hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
                
        self.deploy = True
        print("Model branches successfully fused into a clean single-path architecture!")

if __name__ == "__main__":
    # Test model shape and print parameters
    model = DSUnet(in_channels=3, num_classes=4, deploy=False)
    x = torch.randn(1, 3, 256, 512)
    model.eval()
    
    # Forward in training mode
    y_train = model(x)
    print(f"Training mode output shape: {y_train.shape}")
    
    # Switch to deploy mode and verify forward
    model.switch_to_deploy()
    y_deploy = model(x)
    print(f"Deploy mode output shape: {y_deploy.shape}")
    
    # Verify outputs match
    diff = torch.max(torch.abs(y_train - y_deploy)).item()
    print(f"Difference between training and deploy outputs: {diff:.8f}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
