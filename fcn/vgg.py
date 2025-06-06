import torch.nn as nn
from torchvision.models import vgg11_bn, VGG11_BN_Weights

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        model = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        features = list(model.features.children())

        self.os_8 = nn.Sequential(*features[0:15])
        self.os_16 = nn.Sequential(*features[15:22])
        self.os_32 = nn.Sequential(*features[22:])
        

    def forward(self, image):
        feat_8 = self.os_8(image)
        feat_16 = self.os_16(feat_8)
        feat_32 = self.os_32(feat_16)

        return feat_8, feat_16, feat_32
        