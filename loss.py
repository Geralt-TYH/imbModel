
import torch
import torch.nn.functional as F
from torch import nn


class ClassificationLoss(nn.Module):

    def __init__(self, input: torch.Tensor,target: torch.Tensor, nclass,weight: torch.Tensor = None):
        super(ClassificationLoss, self).__init__()
        self.input = input
        self.target = target
        self.nclass = nclass
        self.weight = weight
    # 分类损失
    def forward(self):
        one_hot = torch.zeros(self.input.shape[0], self.nclass, requires_grad=False)
        if (self.weight != None):
            one_hot = one_hot.scatter(dim=1, index=self.target.unsqueeze(dim=-1), src=self.weight.unsqueeze(dim=-1))
        else:
            one_hot = one_hot.scatter(dim=1, index=self.target.unsqueeze(dim=-1), value=1)

        loss = -torch.mean(one_hot * F.log_softmax(self.input, dim=1))
        return loss