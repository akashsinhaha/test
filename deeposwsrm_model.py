"""
DeepOSWSRM: DEEP feature collaborative CNN for Water Super-Resolution Mapping
based on Optical and SAR images

Implementation of the method from:
"Super-resolution water body mapping with a feature collaborative CNN model 
by fusing Sentinel-1 and Sentinel-2 images"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class StackedResidualCNN(nn.Module):
    """Stacked Residual CNN for feature extraction"""
    def __init__(self, in_channels, base_channels=64, num_blocks=5):
        super(StackedResidualCNN, self).__init__()
        self.conv_in = ConvBlock(in_channels, base_channels)
        
        # Create residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = block(x)
        return x


class SpatialChannelAttention(nn.Module):
    """Combined Spatial and Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x


class WaterFractionUnmixing(nn.Module):
    """Water Fraction Unmixing Module using Pseudo-Siamese CNN"""
    def __init__(self, sentinel1_channels=2, sentinel2_channels=4, base_channels=64):
        super(WaterFractionUnmixing, self).__init__()
        
        # Sentinel-1 feature extractor (SAR)
        self.s1_extractor = StackedResidualCNN(sentinel1_channels, base_channels)
        
        # Sentinel-2 feature extractor (Optical)
        self.s2_extractor = StackedResidualCNN(sentinel2_channels, base_channels)
        
        # Feature fusion network
        self.fusion_network = StackedResidualCNN(base_channels * 2, base_channels)
        
        # Final fraction prediction
        self.fraction_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, sentinel1, sentinel2):
        # Extract features from both sensors
        s1_features = self.s1_extractor(sentinel1)
        s2_features = self.s2_extractor(sentinel2)
        
        # Concatenate features
        fused_features = torch.cat([s1_features, s2_features], dim=1)
        
        # Process fused features
        fused_features = self.fusion_network(fused_features)
        
        # Predict water fraction with custom activation [1 + tanh(x)] / 2
        fraction = self.fraction_conv(fused_features)
        fraction = (1 + torch.tanh(fraction)) / 2
        
        return fraction


class EncoderBlock(nn.Module):
    """Encoder block with convolution and max pooling"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block with transpose convolution and skip connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Add 1x1 conv to adjust channels after concatenation
        self.conv_adjust = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        self.conv_block = ResidualBlock(out_channels)
    
    def forward(self, x, skip):
        x = self.transpose_conv(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        # Adjust channels to out_channels
        x = self.conv_adjust(x)
        # Apply residual block
        x = self.conv_block(x)
        return x


class SuperResolutionMapping(nn.Module):
    """Super-Resolution Water Body Mapping Module with Encoder-Decoder"""
    def __init__(self, in_channels=1, base_channels=64, scale_factor=4):
        super(SuperResolutionMapping, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_in = ConvBlock(in_channels, base_channels)
        
        # Encoder path
        self.encoder1 = EncoderBlock(base_channels, base_channels * 2)
        self.encoder2 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.encoder3 = EncoderBlock(base_channels * 4, base_channels * 8)
        self.encoder4 = EncoderBlock(base_channels * 8, base_channels * 16)
        
        # Bottleneck with attention
        self.bottleneck = ResidualBlock(base_channels * 16)
        self.attention = SpatialChannelAttention(base_channels * 16)
        
        # Decoder path
        # FIXED - Correct skip channel sizes
        self.decoder1 = DecoderBlock(base_channels * 16, base_channels * 16, base_channels * 8)
        self.decoder2 = DecoderBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.decoder3 = DecoderBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.decoder4 = DecoderBlock(base_channels * 2, base_channels * 2, base_channels)
        
        # Multi-scale feature fusion
        # Total channels: dec1(8x) + dec2(4x) + dec3(2x) + dec4(1x) = 8+4+2+1 = 15
        self.fusion_conv = nn.Conv2d(base_channels * (8+4+2+1), base_channels, kernel_size=1)
                
        # Final upsampling and classification
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(base_channels, 2, kernel_size=1)  # 2 classes: water, non-water
    
    def forward(self, fraction):
        # Upsample fraction to target resolution
        fraction_upsampled = self.upsample(fraction)
        
        # Initial features
        x = self.conv_in(fraction_upsampled)
        
        # Encoder
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)
        
        # Bottleneck with attention
        x = self.bottleneck(x)
        x = self.attention(x)
        
        # Decoder with skip connections
        x = self.decoder1(x, skip4)
        dec1 = x
        
        x = self.decoder2(x, skip3)
        dec2 = x
        
        x = self.decoder3(x, skip2)
        dec3 = x
        
        x = self.decoder4(x, skip1)
        dec4 = x
        
        # Multi-scale feature fusion
        dec1_up = F.interpolate(dec1, size=dec4.shape[2:], mode='bilinear', align_corners=True)
        dec2_up = F.interpolate(dec2, size=dec4.shape[2:], mode='bilinear', align_corners=True)
        dec3_up = F.interpolate(dec3, size=dec4.shape[2:], mode='bilinear', align_corners=True)
        
        multi_scale = torch.cat([dec1_up, dec2_up, dec3_up, dec4], dim=1)
        fused = self.fusion_conv(multi_scale)
        
        # Final classification with softmax
        output = self.final_conv(fused)
        
        return output


class DeepOSWSRM(nn.Module):
    """
    Complete DeepOSWSRM Model
    
    Args:
        sentinel1_channels: Number of Sentinel-1 channels (default: 2 for VV and VH)
        sentinel2_channels: Number of Sentinel-2 channels (default: 4 for B, G, R, NIR)
        scale_factor: Super-resolution scale factor (2, 4, or 6)
        base_channels: Base number of channels in the network
    """
    def __init__(self, sentinel1_channels=2, sentinel2_channels=4, scale_factor=4, base_channels=64):
        super(DeepOSWSRM, self).__init__()
        self.scale_factor = scale_factor
        
        # Water fraction unmixing module
        self.unmixing = WaterFractionUnmixing(
            sentinel1_channels=sentinel1_channels,
            sentinel2_channels=sentinel2_channels,
            base_channels=base_channels
        )
        
        # Super-resolution mapping module
        self.srm = SuperResolutionMapping(
            in_channels=1,
            base_channels=base_channels,
            scale_factor=scale_factor
        )
    
    def forward(self, sentinel1, sentinel2):
        """
        Forward pass
        
        Args:
            sentinel1: Sentinel-1 image tensor [B, C, H, W]
            sentinel2: Sentinel-2 image tensor [B, C, H, W] (may contain masked areas)
        
        Returns:
            water_fraction: Coarse-resolution water fraction map [B, 1, H, W]
            water_map: Fine-resolution water body map [B, 2, H*scale, W*scale]
        """
        # Step 1: Estimate water fraction
        water_fraction = self.unmixing(sentinel1, sentinel2)
        
        # Step 2: Super-resolution mapping
        water_map = self.srm(water_fraction)
        
        return water_fraction, water_map
    
    def predict(self, sentinel1, sentinel2):
        """
        Prediction with softmax activation
        
        Returns:
            water_fraction: Water fraction map
            water_map_probs: Probability map for water class
            water_map_binary: Binary water map
        """
        self.eval()
        with torch.no_grad():
            water_fraction, water_map_logits = self.forward(sentinel1, sentinel2)
            water_map_probs = F.softmax(water_map_logits, dim=1)
            water_map_binary = torch.argmax(water_map_probs, dim=1, keepdim=True)
        
        return water_fraction, water_map_probs[:, 1:2], water_map_binary


# Loss functions
class AdaptiveFractionCrossEntropyLoss(nn.Module):
    """
    Adaptive fraction-based cross-entropy loss with class balancing
    """
    def __init__(self, eta=-0.5, water_weight=1.2):
        super(AdaptiveFractionCrossEntropyLoss, self).__init__()
        self.eta = eta
        self.water_weight = water_weight
    
    def forward(self, predictions, targets, fractions):
        """
        Args:
            predictions: Model predictions [B, 2, H, W]
            targets: Ground truth binary maps [B, 1, H, W]
            fractions: Water fraction values [B, 1, H, W]
        """
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        water_prob = probs[:, 1:2]
        
        # Adaptive weight based on fraction
        adaptive_weight = torch.exp(self.eta * fractions)
        
        # Class weight (heavily emphasize water pixels)
        targets = targets.float()
        class_weight = torch.where(targets > 0.5, 
                                   torch.tensor(self.water_weight).to(targets.device),
                                   torch.tensor(1.0).to(targets.device))
        
        # Combined weight
        total_weight = adaptive_weight * class_weight
        
        # Binary cross-entropy
        loss = -total_weight * (
            targets * torch.log(water_prob + 1e-7) + 
            (1 - targets) * torch.log(1 - water_prob + 1e-7)
        )
        
        return loss.mean()

class DeepOSWSRMLoss(nn.Module):
    """
    Combined loss function for DeepOSWSRM
    L = L_frac + λ * L_SRM
    """
    def __init__(self, lambda_weight=1.0, eta=-0.5):
        super(DeepOSWSRMLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.mse_loss = nn.MSELoss()
        self.adaptive_ce_loss = AdaptiveFractionCrossEntropyLoss(eta=eta)
    
    def forward(self, pred_fraction, pred_map, target_fraction, target_map):
        """
        Args:
            pred_fraction: Predicted water fraction [B, 1, H, W]
            pred_map: Predicted fine-resolution water map [B, 2, H', W']
            target_fraction: Target water fraction [B, 1, H, W]
            target_map: Target fine-resolution water map [B, 1, H', W']
        """
        # Fraction loss (MSE)
        loss_frac = self.mse_loss(pred_fraction, target_fraction)
        
        # SRM loss (Adaptive CE)
        loss_srm = self.adaptive_ce_loss(pred_map, target_map, 
                                         F.interpolate(pred_fraction, 
                                                      size=target_map.shape[2:],
                                                      mode='bilinear',
                                                      align_corners=True))
        
        # Combined loss
        total_loss = loss_frac + self.lambda_weight * loss_srm
        
        return total_loss, loss_frac, loss_srm


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with scale factor 4
    model = DeepOSWSRM(
        sentinel1_channels=2,
        sentinel2_channels=4,
        scale_factor=4,
        base_channels=64
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    height, width = 64, 64
    
    sentinel1 = torch.randn(batch_size, 2, height, width).to(device)
    sentinel2 = torch.randn(batch_size, 4, height, width).to(device)
    
    water_fraction, water_map = model(sentinel1, sentinel2)
    
    print(f"\nInput shapes:")
    print(f"  Sentinel-1: {sentinel1.shape}")
    print(f"  Sentinel-2: {sentinel2.shape}")
    print(f"\nOutput shapes:")
    print(f"  Water fraction: {water_fraction.shape}")
    print(f"  Water map: {water_map.shape}")
    
    # Test loss function
    target_fraction = torch.rand(batch_size, 1, height, width).to(device)
    target_map = torch.randint(0, 2, (batch_size, 1, height*4, width*4)).to(device)
    
    loss_fn = DeepOSWSRMLoss()
    total_loss, loss_frac, loss_srm = loss_fn(water_fraction, water_map, target_fraction, target_map)
    
    print(f"\nLoss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Fraction loss: {loss_frac.item():.4f}")
    print(f"  SRM loss: {loss_srm.item():.4f}")
