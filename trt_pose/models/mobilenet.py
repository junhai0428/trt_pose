import torch
import torchvision
from .common import *


class MobileNetBackbone(torch.nn.Module):
    
    def __init__(self, mobilenet):
        super(MobileNetBackbone, self).__init__()
        self.mobilenet = mobilenet
    
    def forward(self, x):
        
        x = self.mobilenet.features(x)

        return x


def _mobilenet_pose(cmap_channels, paf_channels, upsample_channels, mobilenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        MobileNetBackbone(mobilenet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    
    
def mobilenet_baseline(cmap_channels, paf_channels, upsample_channels=256, num_upsample=3, num_flat=0):
    mobilenet = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights)
    return _mobilenet_pose(cmap_channels, paf_channels, upsample_channels, mobilenet, 576, num_upsample, num_flat)
  
    
def _mobilenet_pose_att(cmap_channels, paf_channels, upsample_channels, mobilenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        MobileNetBackbone(mobilenet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

    
def mobilenet_baseline_att(cmap_channels, paf_channels, upsample_channels=256, num_upsample=3, num_flat=0):
    mobilenet = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights)
    return _mobilenet_pose_att(cmap_channels, paf_channels, upsample_channels, mobilenet, 576, num_upsample, num_flat)
