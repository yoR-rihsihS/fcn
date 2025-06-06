import torch.nn as nn

from .vgg import VGG

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.vgg = VGG()

        self.score_pool_32 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.score_pool_16 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.score_pool_8 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)

        self.upscore_32   = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_16   = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_8   = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, image):
        feat_8, feat_16, feat_32 = self.vgg(image)

        score_32 = self.score_pool_32(feat_32)
        score_16 = self.score_pool_16(feat_16)
        score_8 = self.score_pool_8(feat_8)

        x = self.upscore_32(score_32)
        x = x + score_16

        x = self.upscore_16(x)
        x = x + score_8

        x = self.upscore_8(x)
        return x

