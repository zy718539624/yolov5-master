import random
import os
import shutil
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def rename():
    dir_path = "F://课题//数据集//留存//2//自己整理//"
    beg_num = 292
    img_list = os.listdir(dir_path)
    img_list.sort()
    for img_file in img_list:
        os.rename(dir_path + img_file,dir_path + "{}.jpg".format(beg_num))
        beg_num+=1

def make_val():
    a  = random.sample(os.listdir("coco128/images/train2017/"), 55)
    for img_name in a:
        src_img = "coco128/images/train2017/"+img_name
        dst_img = "coco128/images/val2017/"+img_name
        shutil.move(src_img, dst_img)

        img_num = img_name.split(".")[0]
        txt_name = img_num + ".txt"
        print(txt_name)
        src_txt = "coco128/labels/train2017/" + txt_name
        dst_txt = "coco128/labels/val2017/" + txt_name
        shutil.move(src_txt, dst_txt)

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5m.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/boll.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default="2,3,5,6", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()
    print(opt.nosave)

def eva_convout():
    input_size = 31

    input0 = torch.rand([1, 3, input_size,input_size])
    conv1 = torch.nn.Conv2d(in_channels=3,
                            out_channels=1,
                            kernel_size=3,
                            stride=2,
                            padding=1,)
    out = conv1(input0)
    print(out.shape)
    # out_size =(input_size-kernel_size+2*padding) / stride + 1
    # print(out_size)

def test_asff():
    import torch
    import torch.nn.functional as F
    from models.common import ASFF
    asff = ASFF(2)
    input0 = torch.rand([1, 768, 20, 20])
    input1 = torch.rand([1, 384, 40, 40])
    input2 = torch.rand([1, 192, 80, 80])
    input = [input0, input1, input2]
    asff(input)

def get_map(num):
    import codecs

    f = codecs.open('runs/exp{}/results.txt'.format(num), mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
    line = f.readline()  # 以行的形式进行读取文件
    list1 = []
    map50_list = []
    map5095_list = []
    while line:
        a = line.split()
        b = a[-7:-3]  # 这是选取需要读取的位数
        map50 = float(b[-2])
        map5095 = float(b[-1])
        map50_list.append(map50)  # 将其添加在列表之中
        map5095_list.append(map5095)
        list1.append(b)
        line = f.readline()
    f.close()

    print("Max_map50:", max(map50_list), "   all:", list1[map50_list.index(max(map50_list))])
    print("Max_map50-95:",max(map5095_list),"   all:",list1[map5095_list.index(max(map5095_list))])

def watch_gpu():
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
                print("第%s号卡" % i,free)
                os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
                a = torch.rand([1,3,500,500])

                a.cuda()
                print_here = True
                while True:
                    if print_here:
                        print("已经完成")
                        print_here = False


            time.sleep(1)



    # total0 = meminfo0.total / 1024 ** 2  # 第n块显卡总的显存大小
    # used0 = meminfo0.used / 1024 ** 2  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    # free0 = meminfo0.free / 1024 ** 2  # 第n块显卡剩余显存大小

def test_conv():
    input0 = torch.rand([1, 3, 80, 80])
    weight = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3, ]])
    conv1 = torch.nn.Conv2d(in_channels=3,
                            out_channels=1,
                            kernel_size=3,
                            stride=2,
                            padding=2,
                            bias=False,
                            dilation=2)

    conv2 = torch.nn.Conv2d(in_channels=3,
                            out_channels=1,
                            kernel_size=3,
                            stride=2,
                            padding=3,
                            bias=False,
                            dilation=3)

    conv2.weight.data = conv1.weight.data
    out = conv2(input0)
    print(out.shape)

def test_activaton():
    x = torch.linspace(-10, 10, 100)
    print(x.shape)
    act = torch.nn.Hardswish()
    y1 = act(x)
    y2 = x * (torch.tanh(F.softplus(x)))
    plt.plot(x, y1, label='Hardswish')
    plt.plot(x, y2, label="mish")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # watch_gpu()
    get_map(63)
    # test_asff()
    # eva_convout()
    # rename()
