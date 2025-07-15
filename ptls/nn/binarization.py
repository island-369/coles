from torch import nn as nn
from torch.autograd import Function


class Binarization(Function):
    @staticmethod
    def forward(self, x):
        q = (x > 0).float()
        return 2*q - 1

    @staticmethod
    def backward(self, grad_outputs):
        return grad_outputs


binary = Binarization.apply


class BinarizationLayer(nn.Module):
    def __init__(self, hs_from, hs_to):
        super(BinarizationLayer, self).__init__()
        self.linear = nn.Linear(hs_from, hs_to, bias=False)

    def forward(self, x):
        # 确保输入数据类型与线性层权重一致，避免混合精度训练时的类型不匹配错误
        # x = x.to(self.linear.weight.dtype)
        return binary(self.linear(x))