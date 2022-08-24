import torch
from torch import nn
from torchvision.models import vgg16
from utils import get_multi_anchor_prior


class SSD(nn.Module):
    def __init__(self):
        super(SSD, self).__init__()
        self.base_network = self.base_net()

    def forward(self, X):
        return self.base_network(X)

    @staticmethod
    def base_net():
        """
        VGG16 contains 3 block:
        VGG16 = Features + avgpoll + classifier
        for this project, I just want to use the features block
        """
        net = vgg16(pretrained=False).features
        return net

    @staticmethod
    def cls_predictor(num_inputs, num_anchors, num_classes):
        return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

    @staticmethod
    def bbox_predictor(num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    @staticmethod
    def small_forward(x, block):
        return block(x)

    @staticmethod
    def flatten_pred(pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

    @classmethod
    def concat_preds(cls, preds):
        return torch.cat([cls.flatten_pred(pred) for pred in preds], dim=1)

    @staticmethod
    def down_sample_blk(in_channels, out_channels):
        blk = []
        for _ in range(2):
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*blk)

    @classmethod
    def get_blk(cls, i):
        if i == 0:
            blk = cls.base_net()
        elif i == 1:
            blk = cls.down_sample_blk(64, 128)
        elif i == 4:
            blk = nn.AdaptiveAvgPool2d((1, 1))
        else:
            blk = cls.down_sample_blk(128, 128)
        return blk

    @staticmethod
    def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        anchors = get_multi_anchor_prior(Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds= bbox_predictor(Y)
        return Y, anchors, cls_preds, bbox_preds



