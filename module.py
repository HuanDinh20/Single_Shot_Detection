import torch
from torch import nn
from torchvision.models import vgg16
from utils import get_multi_anchor_prior


class SSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(SSD, self).__init__()
        self.base_network = self.base_net()
        self.net = nn.Sequential()
        self.num_classes = num_classes
        self.sizes = [[0.2, 0.272],
                      [0.37, 0.447],
                      [0.54, 0.619],
                      [0.71, 0.79],
                      [0.88, 0.961]]

        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        for i in range(5):
            setattr(self, f"blk_{i}", self.get_blk(i))
            setattr(self, f'cls_{i}', self.cls_predictor(num_classes=self.num_classes,
                                                         num_anchors=self.num_anchors))
            setattr(self, f'bbox_{i}', self.bbox_predictor(self.num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                X, getattr(self, f"blk_{i}"), self.sizes[i], self.ratios[i],
                getattr(self, f"cls_{i}"), getattr(self, f"bbox_{i}"))

        anchors = torch.cat(anchors, dim=1)
        cls_preds = self.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = self.concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

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
    def cls_predictor(num_anchors, num_classes):
        return nn.LazyConv2d(num_anchors * (num_classes + 1), kernel_size=3, padding=1)

    @staticmethod
    def bbox_predictor(num_anchors):
        return nn.LazyConv2d(num_anchors * 4, kernel_size=3, padding=1)

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
    def down_sample_blk(out_channels):
        blk = []
        for _ in range(2):
            blk.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
        blk.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*blk)

    @classmethod
    def get_blk(cls, i):
        if i == 0:
            blk = cls.base_net()
        elif i == 1:
            blk = cls.down_sample_blk(128)
        elif i == 4:
            blk = nn.AdaptiveAvgPool2d((1, 1))
        else:
            blk = cls.down_sample_blk(128)
        return blk

    @staticmethod
    def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        anchors = get_multi_anchor_prior(Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return Y, anchors, cls_preds, bbox_preds
