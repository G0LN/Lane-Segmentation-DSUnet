import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for Channel Attention.
    Compresses spatial dimensions via Global Average Pooling, then applies
    a two-layer MLP bottleneck (C -> C/reduction -> C) to predict channel weights,
    mapping them to [0, 1] via Sigmoid to recalibrate the input features.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        hidden_channels = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights

class RepDSConv(nn.Module):
    """
    Reparameterizable Depthwise Separable Convolution block with SE Attention.
    During training: multi-branch 7x7 DW + 3x3 DW + 1x1 DW + identity.
    During inference: fused single 7x7 DW + standard pointwise conv + SE Attention.
    """
    def __init__(self, in_channels, out_channels, deploy=False):
        super(RepDSConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy

        # Pointwise convolution is always 1x1 Conv + BN + ReLU
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU(inplace=True)

        # Squeeze-and-Excitation Attention Block
        self.se = SEBlock(out_channels, reduction=16)

        if deploy:
            self.dw_reparam = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=True)
            self.relu_dw = nn.ReLU(inplace=True)
        else:
            # Branch 1: 7x7 Depthwise Conv + BN
            self.dw_7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False)
            self.bn_7x7 = nn.BatchNorm2d(in_channels)

            # Branch 2: 3x3 Depthwise Conv + BN
            self.dw_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(in_channels)

            # Branch 3: 1x1 Depthwise Conv + BN
            self.dw_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, groups=in_channels, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(in_channels)

            # Branch 4: Identity (Only BN)
            self.bn_identity = nn.BatchNorm2d(in_channels)
            self.relu_dw = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            x = self.dw_reparam(x)
            x = self.relu_dw(x)
        else:
            x_7x7 = self.bn_7x7(self.dw_7x7(x))
            x_3x3 = self.bn_3x3(self.dw_3x3(x))
            x_1x1 = self.bn_1x1(self.dw_1x1(x))
            x_id = self.bn_identity(x)
            x = x_7x7 + x_3x3 + x_1x1 + x_id
            x = self.relu_dw(x)
            
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu_pw(x)
        x = self.se(x)
        return x

    def switch_to_deploy(self):
        if self.deploy:
            return
        
        fused_kernel, fused_bias = self.get_equivalent_kernel_bias()
        
        # Instantiate the single fused 7x7 DW convolution
        self.dw_reparam = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=7, padding=3, groups=self.in_channels, bias=True)
        self.dw_reparam.weight.data = fused_kernel
        self.dw_reparam.bias.data = fused_bias
        
        # Delete training-only branches to free GPU/RAM memory
        del self.dw_7x7
        del self.bn_7x7
        del self.dw_3x3
        del self.bn_3x3
        del self.dw_1x1
        del self.bn_1x1
        del self.bn_identity
        
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        # 1. Fuse the 7x7 Conv + BN branch
        kernel_7x7, bias_7x7 = self._fuse_conv_bn(self.dw_7x7, self.bn_7x7)

        # 2. Fuse the 3x3 Conv + BN branch and pad kernel to 7x7
        kernel_3x3_raw, bias_3x3 = self._fuse_conv_bn(self.dw_3x3, self.bn_3x3)
        kernel_3x3 = F.pad(kernel_3x3_raw, [2, 2, 2, 2])

        # 3. Fuse the 1x1 Conv + BN branch and pad kernel to 7x7
        kernel_1x1_raw, bias_1x1 = self._fuse_conv_bn(self.dw_1x1, self.bn_1x1)
        kernel_1x1 = F.pad(kernel_1x1_raw, [3, 3, 3, 3])

        # 4. Fuse the Identity (BN only) branch
        # Create identity 7x7 depthwise kernel: shape [in_channels, 1, 7, 7]
        device = self.bn_identity.weight.device
        id_kernel = torch.zeros((self.in_channels, 1, 7, 7), device=device)
        for i in range(self.in_channels):
            id_kernel[i, 0, 3, 3] = 1.0
            
        kernel_id, bias_id = self._fuse_conv_bn_raw(id_kernel, None, self.bn_identity)

        # 5. Sum up all fused branches
        fused_kernel = kernel_7x7 + kernel_3x3 + kernel_1x1 + kernel_id
        fused_bias = bias_7x7 + bias_3x3 + bias_1x1 + bias_id
        return fused_kernel, fused_bias

    def _fuse_conv_bn(self, conv, bn):
        return self._fuse_conv_bn_raw(conv.weight, conv.bias, bn)

    def _fuse_conv_bn_raw(self, kernel, bias, bn):
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)

        fused_kernel = kernel * t
        
        if bias is not None:
            fused_bias = (bias - mean) * (gamma / std) + beta
        else:
            fused_bias = (-mean) * (gamma / std) + beta
            
        return fused_kernel, fused_bias


# Maintain standard DSConv class for legacy compatibility if needed elsewhere
class DSConv(nn.Module):
    """Depthwise Separable Convolution Block (Legacy)"""
    def __init__(self, in_channels, out_channels):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True, dropout_prob=0.0, deploy=False):
        super(EncoderBlock, self).__init__()
        self.use_pool = use_pool
        # DSUnet uses two RepDSConv layers per block
        self.conv = nn.Sequential(
            RepDSConv(in_channels, out_channels, deploy=deploy),
            RepDSConv(out_channels, out_channels, deploy=deploy)
        )
        
        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        features = self.conv(x)
        features = self.dropout(features)
        
        if self.use_pool:
            pooled = self.pool(features)
            return features, pooled
        else:
            return features

    def switch_to_deploy(self):
        for layer in self.conv:
            if hasattr(layer, 'switch_to_deploy'):
                layer.switch_to_deploy()
