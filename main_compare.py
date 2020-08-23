import time
import argparse
import numpy as np
import torch
from resnet import resnet18
from torch2trt import TRTModule


parser = argparse.ArgumentParser(description='A example for TensorRT and Torch2TRT')
parser.add_argument('--path_ckpt_pytorch', type=str, default='./resnet18-5c106cde.pth', help='Path for a pretrained PyTorch weights.')
parser.add_argument('--path_ckpt_trt', type=str, default='./resnet18-trt.pth', help='Path for a TRT weights.')
parser.add_argument('--num', type=int, default=10, help='A number of test')
parser.add_argument('--batch', type=int, default=1, help='A batch of an input tensor')
parser.add_argument('--channel', type=int, default=3, help='A channel of an input tensor')
parser.add_argument('--height', type=int, default=224, help='A height of an input tensor')
parser.add_argument('--width', type=int, default=224, help='A width of an input tensor')
args = parser.parse_args()


class AverageMeter_time:
    def __init__(self):
        self.list_time = []
        self.list_val = []

    def tic(self):
        self.time_start = time.time()

    def toc(self):
        self.time_end = time.time()
        self._update_time()

    def rec(self, _ndarr):
        self.list_val.append(_ndarr)

    def _mean(self, _list):
        if len(_list) == 0:
            raise ZeroDivisionError

        return sum(_list) / len(_list)

    def _update_time(self):
        self.list_time.append(self.time_end - self.time_start)
        self.tot_time = sum(self.list_time)
        self.avg_time = self._mean(_list=self.list_time)
        self.max_time = max(self.list_time)
        self.min_time = min(self.list_time)


class AverageMeter_value:
    def __init__(self):
        self.list_avg = []
        self.list_max = []
        self.list_min = []

    def rec(self, _ndarr):
        self.list_avg.append(_ndarr.mean())
        self.list_max.append(_ndarr.max())
        self.list_min.append(_ndarr.min())


if __name__ == '__main__':
    time_preparation_tic = time.time()
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    avgmeter_pytorch = AverageMeter_time()
    avgmeter_trt = AverageMeter_time()
    avgmeter_error = AverageMeter_value()
    time_preparation_toc = time.time()

    time_model_pytorch_tic = time.time()
    model_pytorch = resnet18().eval().to(device)
    model_pytorch.load_state_dict(torch.load(args.path_ckpt_pytorch))
    x = torch.rand((args.batch, args.channel, args.height, args.width)).to(device)
    time_model_pytorch_toc = time.time()

    time_model_trt_tic = time.time()
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.path_ckpt_trt))
    time_model_trt_toc = time.time()

    time_computation_tic = time.time()
    for idx in range(args.num):
        x = torch.rand((args.batch, args.channel, args.height, args.width)).to(device)

        avgmeter_pytorch.tic()
        y_pytorch = model_pytorch(x)
        avgmeter_pytorch.toc()

        avgmeter_trt.tic()
        y_trt = model_trt(x)
        avgmeter_trt.toc()

        ndarr_y_pytroch = y_pytorch.data.cpu().numpy()
        ndarr_y_trt = y_trt.data.cpu().numpy()

        avgmeter_pytorch.rec(_ndarr=ndarr_y_pytroch)
        avgmeter_trt.rec(_ndarr=ndarr_y_trt)
        avgmeter_error.rec(_ndarr=np.abs(ndarr_y_pytroch - ndarr_y_trt))
    time_computation_toc = time.time()

    print('Time profile [sec.]')
    print('i)   Total:                {:.3f}'.format(time_computation_toc - time_preparation_tic))
    print('ii)  Preparation:          {:.3f}'.format(time_preparation_toc - time_preparation_tic))
    print('iii) Load model (PyTorch): {:.3f}'.format(time_model_pytorch_toc - time_model_pytorch_tic))
    print('iv)  Load model (TRT):     {:.3f}'.format(time_model_trt_toc - time_model_trt_tic))
    print('v)   Computation:          {:.3f}'.format(time_computation_toc - time_computation_tic))

    for idx in range(args.num):
        # The error means maximum absolute difference.
        # When you want to check the Mean Absolute Difference (MAE), you can check the variable, avgmeter_error.list_avg[idx].
        print('[{:d}/{:d}] Time: PyTorch: {:.3e}, TRT: {:.3e} / Error: {:.3e}'.format(idx, args.num - 1, avgmeter_pytorch.list_time[idx], avgmeter_trt.list_time[idx], avgmeter_error.list_max[idx]))
