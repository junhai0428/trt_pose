import torch
import torchvision
from .common import *


class ShuffleNetBackbone(torch.nn.Module):
    
    def __init__(self, shufflenet):
        super(ShuffleNetBackbone, self).__init__()
        self.shufflenet = shufflenet
    
    def forward(self, x):
        x = self.shufflenet.conv1(x)
        x = self.shufflenet.maxpool(x)
        x = self.shufflenet.stage2(x)
        x = self.shufflenet.stage3(x)
        x = self.shufflenet.stage4(x)
        x = self.shufflenet.conv5(x)
        return x


def _shufflenet_pose(cmap_channels, paf_channels, upsample_channels, shufflenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ShuffleNetBackbone(shufflenet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    
    
def shufflenet_baseline(cmap_channels, paf_channels, upsample_channels=256, num_upsample=3, num_flat=0):
    shufflenet = torchvision.models.shufflenet_v2_x0_5(weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    return _shufflenet_pose(cmap_channels, paf_channels, upsample_channels, shufflenet, 1024, num_upsample, num_flat)
  
    
def _shufflenet_pose_att(cmap_channels, paf_channels, upsample_channels, shufflenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ShuffleNetBackbone(shufflenet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

    
def shufflenet_baseline_att(cmap_channels, paf_channels, upsample_channels=256, num_upsample=3, num_flat=0):
    shufflenet = torchvision.models.shufflenet_v2_x0_5(weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    return _shufflenet_pose_att(cmap_channels, paf_channels, upsample_channels, shufflenet, 1024, num_upsample, num_flat)
