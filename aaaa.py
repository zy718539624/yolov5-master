import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, c1,growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(c1))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(c1, bn_size*growth_rate,kernel_size=1, stride=1, bias=False))

        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = 0.2

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, c1,growth_rate, bn_size):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(c1+i*growth_rate,growth_rate, bn_size)
            self.add_module("denselayer%d" % (i+1,), layer)
        self.add_module("bn_dense",nn.BatchNorm2d(c1+num_layers*growth_rate))
        self.add_module("relu_dense", nn.ReLU(inplace=True))
        self.add_module("conv_dense",nn.Conv2d(c1+num_layers*growth_rate, c1,kernel_size=1, stride=1))

class GPUModel(nn.Module):
    def __init__(self,numbers):
        super(GPUModel,self).__init__()
        self.m = nn.Sequential(*[_DenseBlock(9,640,32,4) for _ in range(numbers)])
    def forward(self,inputs):
        out= self.m(inputs)

def watch_gpu(k):
    import pynvml
    import time
    import torch
    pynvml.nvmlInit()
    gpu_num = [0,1,2,3,4,5,6,7]
    while True:
        for i in gpu_num:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free = meminfo.free / 1024 ** 2

            if free >= 2000:
                print("第%s号卡存在剩余空间： " % i, free)
                os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
                # a = torch.rand([1,3,500,500])
                from models.common import GPUModel
                model = GPUModel(120 * k)
                model.cuda()
                print_here = True
                while True:
                    if print_here:
                        print("已经完成")
                        print_here = False

            time.sleep(1)

    # total0 = meminfo0.total / 1024 ** 2  # 第n块显卡总的显存大小
    # used0 = meminfo0.used / 1024 ** 2  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    # free0 = meminfo0.free / 1024 ** 2  # 第n块显卡剩余显存大小

if __name__ == '__main__':

    watch_gpu(3)