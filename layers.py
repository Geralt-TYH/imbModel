import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        #it's a sparse matric mutiply process
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GraphConvolution(in_channels, 2 * out_channels)
        self.conv2 = GraphConvolution(2 * out_channels, out_channels)
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = F.relu(self.conv1(input, adj))
        return self.conv2(input, adj)

class InnerProductDecoder(Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()
    def forward(self, z, sigmoid=True):
        adj = torch.mm(z,z.t())
        return torch.sigmoid(adj) if sigmoid else adj

