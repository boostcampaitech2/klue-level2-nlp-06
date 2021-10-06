import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
        def __init__(self, weight=None,
                    gamma=2.5, alpha = 0.25, reduction='mean'):
            nn.Module.__init__(self)
            self.weight = weight
            self.gamma = gamma
            self.reduction = reduction
            self.alpha = alpha

        def forward(self, input_tensor, target_tensor):
            log_prob = F.log_softmax(input_tensor, dim=-1)
            prob = torch.exp(log_prob)
            return F.nll_loss(
                (self.alpha * (1 - prob) ** self.gamma) * log_prob,
                target_tensor,
                weight=self.weight,
                reduction=self.reduction
            )
