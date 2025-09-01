import torch
import torch.nn as nn

class CrossEntropy2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: (B, 1, H, W), target: (B, H, W)
        pred = pred.squeeze(1)
        loss = -(target * torch.log(pred + 1e-10)).sum(dim=(1,2)).mean()
        return loss