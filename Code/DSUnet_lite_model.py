import torch
import torch.nn as nn

# --- 0. DEPLOYABLE MODEL (Wrapper) ---
class DeployableDSUnet(nn.Module):
    def __init__(self, core_model, target_size=(288, 512)):
        super(DeployableDSUnet, self).__init__()
        self.core_model = core_model
        self.target_h = target_size[0]
        self.target_w = target_size[1]
        self.register_buffer('scale', torch.tensor(255.0))

    def forward(self, x):
        # Permute from [B, H, W, C] to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = x.float() / self.scale
        
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = torch.nn.functional.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        
        logits = self.core_model(x)
        return torch.softmax(logits, dim=1)

# --- 1. CORE BLOCKS ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DSUnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSUnetBlock, self).__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class StandardBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# --- 2. MAIN ARCHITECTURE (LITE VERSION) ---
class DSUnet(nn.Module):
    def __init__(self, n_classes):
        super(DSUnet, self).__init__()
        # Encoder 1 (Reduced: 64 -> 32)
        self.c1 = StandardBlock(3, 32)
        self.p1 = nn.MaxPool2d(2)
        
        # Encoder 2 (Reduced: 128 -> 64)
        self.c2 = DSUnetBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)
        
        # Encoder 3 (Reduced: 256 -> 128)
        self.c3 = DSUnetBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)
        
        # Encoder 4 (Reduced: 512 -> 256)
        self.c4 = DSUnetBlock(128, 256)
        self.drop4 = nn.Dropout(0.25)
        self.p4 = nn.MaxPool2d(2)
        
        # Bottleneck (Reduced: 1024 -> 512)
        self.b = DSUnetBlock(256, 512)
        self.drop_b = nn.Dropout(0.25)
        
        # Decoder
        # Up 6 (Input 512 -> Output 256)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c6 = DSUnetBlock(512, 256) # Concatenation: 256 (up) + 256 (skip) = 512 in
        
        # Up 7 (Input 256 -> Output 128)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c7 = DSUnetBlock(256, 128) # Concatenation: 128 (up) + 128 (skip) = 256 in
        
        # Up 8 (Input 128 -> Output 64)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c8 = DSUnetBlock(128, 64) # Concatenation: 64 (up) + 64 (skip) = 128 in
        
        # Up 9 (Input 64 -> Output 32)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c9 = DSUnetBlock(64, 32) # Concatenation: 32 (up) + 32 (skip) = 64 in
        
        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        
        # Split Flow at Layer 4 (now 256 channels)
        feat4 = self.c4(self.p3(c3)) 
        p4 = self.p4(feat4)         # To Bottleneck (No Dropout)
        c4_skip = self.drop4(feat4) # To Decoder (With Dropout)
        
        # Bottleneck
        b = self.drop_b(self.b(p4))
        
        # Decoder
        u6 = self.up6(b)
        if u6.size() != c4_skip.size():
            u6 = torch.nn.functional.interpolate(u6, size=c4_skip.shape[2:], mode='bilinear', align_corners=False)   
        u6 = self.c6(torch.cat([u6, c4_skip], dim=1))

        u7 = self.up7(u6)
        if u7.size() != c3.size():
            u7 = torch.nn.functional.interpolate(u7, size=c3.shape[2:], mode='bilinear', align_corners=False)
        u7 = self.c7(torch.cat([u7, c3], dim=1))
        
        u8 = self.up8(u7)
        if u8.size() != c2.size():
            u8 = torch.nn.functional.interpolate(u8, size=c2.shape[2:], mode='bilinear', align_corners=False)
        u8 = self.c8(torch.cat([u8, c2], dim=1))
        
        u9 = self.up9(u8)
        if u9.size() != c1.size():
            u9 = torch.nn.functional.interpolate(u9, size=c1.shape[2:], mode='bilinear', align_corners=False)
        u9 = self.c9(torch.cat([u9, c1], dim=1))
        
        return self.out(u9)