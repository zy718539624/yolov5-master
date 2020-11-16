import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pynvml
import time
import torch


class GPUModel(nn.Module):
    def __init__(self):
        super(GPUModel, self).__init__()
        self.m = nn.Conv2d(1, 1, 1)

    def forward(self, inputs):
        out = self.m(inputs)
        return out


def get_free_me(i):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free = meminfo.free / 1024 ** 2
    return free


def watch_gpu():
    pynvml.nvmlInit()
    gpu_num = [0, 3, 5]
    print("watching")
    while True:
        for i in gpu_num:
            free = get_free_me(i)
            if free >= 5000:
                print("第{}号卡存在剩余空间：".format(i), free)
                os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
                try:
                    model = GPUModel()
                    model.cuda()
                    tensor_a = torch.rand([6, 1, 10000, 10000]).cuda()
                    b = model(tensor_a)
                    print("抢到并继续")
                except RuntimeError:
                    print('没抢到！')
                while True:
                    free2 = get_free_me(i)
                    if free2 > 1000:
                        print("发现剩余空间：", free2)
                        try:
                            tensor_b = torch.rand([2, 1, 10000, 10000]).cuda()
                            # model = GPUModel()
                            # model.cuda()
                            c = model(tensor_b)
                            print("+2000")
                        except RuntimeError:
                            print('没抢到！')
            #     print_here = True
            #     try:
            #         if print_here:
            #             print("已经完成")
            #             print_here = False
            #         else:
            #             free2 = get_free_me(i)
            #             if free2 > 1400:
            #                 print("发现剩余空间：", free2)
            #                 k = math.floor(free / 1024)
            #                 model = GPUModel(120 * k)
            #                 model.cuda()
            #     except RuntimeError:
            #         print('Uh oh')
            #         # time.sleep(2)
            #
            # time.sleep(1)

    # total0 = meminfo0.total / 1024 ** 2  # 第n块显卡总的显存大小
    # used0 = meminfo0.used / 1024 ** 2  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    # free0 = meminfo0.free / 1024 ** 2  # 第n块显卡剩余显存大小


if __name__ == '__main__':
    watch_gpu()
