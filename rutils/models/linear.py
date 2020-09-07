"""
implement my own linear layer which support gradient control for each individual dim
"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init

import math

class Linear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_list = [Parameter(torch.Tensor(1, out_features)) for _ in range(in_features)]
        self.weight_param_list = torch.nn.ParameterList(self.weight_list)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.weight_list:
          init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = torch.cat(self.weight_list, 0).t()
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )