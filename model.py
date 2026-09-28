"""
EfficientNet-V2-S model adaptation for 4-channel input along with Focal Loss with label smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from config import NUM_CLASSES, DROPOUT

def build_model():
    """Modifies EfficientNet-V2-S stem to accept 4 channels and attaches an MLP classification head."""
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None)
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True) * 0.5
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.features[0][0] = new_conv
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=DROPOUT * 0.8),
        nn.Linear(256, NUM_CLASSES)
    )
    return model

class FocalLabelSmoothingCE(nn.Module):
    """Combines multiclass focal loss modulation with label smoothing regularization."""
    def __init__(self, weights, smoothing=0.05, alpha=0.25, gamma=2.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        ce_loss = -(true_dist * log_probs).sum(dim=1)
        probs = torch.softmax(logits, dim=1)
        p_t = (true_dist * probs).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss
        return (loss * self.weights[targets]).mean()
