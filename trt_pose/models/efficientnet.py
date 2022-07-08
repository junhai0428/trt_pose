import torch
import torchvision
from .common import *


class EfficientNetBackbone(torch.nn.Module):
    
    def __init__(self, efficientnet):
        super(EfficientNetBackbone, self).__init__()
        self.efficientnet = efficientnet
    
    def forward(self, x):
        
        x = self.efficientnet.features(x)

        return x


def _efficientnet_pose(cmap_channels, paf_channels, upsample_channels, efficientnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        EfficientNetBackbone(efficientnet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    
    
def efficientnet_baseline(cmap_channels, paf_channels, upsample_channels=256, num_upsample=3, num_flat=0):
    efficientnet = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
    return _efficientnet_pose(cmap_channels, paf_channels, upsample_channels, efficientnet, 960, num_upsample, num_flat)
  
    
def _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        EfficientNetBackbone(efficientnet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

    
def efficientnet_baseline_att(cmap_channels, paf_channels, upsample_channels=256, num_upsample=3, num_flat=0):
    efficientnet = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
    return _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, 960, num_upsample, num_flat)
