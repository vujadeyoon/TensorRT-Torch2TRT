import os
import time
import argparse
from tqdm import tqdm
import numpy as np

import statistics
import torch
from resnet import resnet18
from torch2trt import TRTModule


parser = argparse.ArgumentParser(description='A example for TensorRT and Torch2TRT')
parser.add_argument('--path_ckpt_pth', type=str, default='./resnet18-5c106cde.pth', help='Path for a pretrained PyTorch weights.')
parser.add_argument('--batch', type=int, default=1, help='A batch of an input tensor')
parser.add_argument('--channel', type=int, default=3, help='A channel of an input tensor')
parser.add_argument('--height', type=int, default=224, help='A height of an input tensor')
parser.add_argument('--width', type=int, default=224, help='A width of an input tensor')
parser.add_argument('--n_times', type=int, default=100, help='The number of iterations')
parser.add_argument('--warmup_time', type=int, default=5, help='The number of warmup')
args = parser.parse_args()


class AverageMeterTimeValue:
    def __init__(self, _warmup_time=0):
        self.warmup_time = _warmup_time
        self.cnt_call = 0
        self.list_time = []
        self.list_val = []

    def tic(self):
        self.time_start = time.time()

    def toc(self):
        self.time_end = time.time()
        self.cnt_call += 1

        if self.warmup_time < self.cnt_call:
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
        self.avg_fps = 1 / self.avg_time
        self.max_time = max(self.list_time)
        self.min_time = min(self.list_time)


class AverageMeterStat:
    def __init__(self):
        self.list_avg = []
        self.list_max = []
        self.list_min = []

    def rec(self, _ndarr):
        self.list_avg.append(_ndarr.mean())
        self.list_max.append(_ndarr.max())
        self.list_min.append(_ndarr.min())


if __name__ == '__main__':
    if args.batch != 1:
        raise ValueError('The args.batch should be 1 because the Torch2TRT does not support multi-batch size.')

    time_preparation_tic = time.time()
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    avgmeter_pth = AverageMeterTimeValue(_warmup_time=args.warmup_time)
    avgmeter_trt = AverageMeterTimeValue(_warmup_time=args.warmup_time)
    avgmeter_gpusynch_pth = AverageMeterTimeValue(_warmup_time=args.warmup_time)
    avgmeter_gpusynch_trt = AverageMeterTimeValue(_warmup_time=args.warmup_time)
    avgmeter_error = AverageMeterStat()
    time_preparation_toc = time.time()

    time_model_pth_tic = time.time()
    model_pth = resnet18().eval().to(device)
    model_pth.load_state_dict(torch.load(args.path_ckpt_pth))
    x = torch.rand((args.batch, args.channel, args.height, args.width)).to(device)
    time_model_pth_toc = time.time()

    time_model_trt_tic = time.time()
    model_trt = TRTModule()

    path_ckpt_trt = './resnet18_{}x{}-trt.pth'.format(args.height, args.width)

    if os.path.isfile(path=path_ckpt_trt) is True:
        model_trt.load_state_dict(torch.load(path_ckpt_trt))
    else:
        raise ValueError('The ckpt file, {}, is not existed.'.format(path_ckpt_trt))

    time_model_trt_toc = time.time()

    time_computation_tic = time.time()
    for idx in tqdm(range(args.n_times)):
        x = torch.rand((args.batch, args.channel, args.height, args.width)).to(device)

        # TRT Network forward
        avgmeter_trt.tic()
        y_trt = model_trt(x)
        avgmeter_trt.toc()
        # TRT GPU Synch.
        avgmeter_gpusynch_trt.tic()
        ndarr_y_trt = y_trt.data.cpu().numpy()
        avgmeter_gpusynch_trt.toc()

        # PyTorch Network forward
        avgmeter_pth.tic()
        y_pth = model_pth(x)
        avgmeter_pth.toc()
        # PyTorch GPU Synch.
        avgmeter_gpusynch_pth.tic()
        ndarr_y_pytroch = y_pth.data.cpu().numpy()
        avgmeter_gpusynch_pth.toc()

        avgmeter_pth.rec(_ndarr=ndarr_y_pytroch)
        avgmeter_trt.rec(_ndarr=ndarr_y_trt)
        avgmeter_error.rec(_ndarr=np.abs(ndarr_y_pytroch - ndarr_y_trt))
    time_computation_toc = time.time()

    time_total = time_computation_toc - time_preparation_tic
    time_preparation = time_preparation_toc - time_preparation_tic
    time_model_pth = time_model_pth_toc - time_model_pth_tic
    time_model_trt = time_model_trt_toc - time_model_trt_tic
    time_computation = time_computation_toc - time_computation_tic

    print('Time profile. unit: sec. [FPS]')
    print('i)   Total:                {:.3f} [{:5.2f}]'.format(time_total, 1 / time_total))
    print('ii)  Preparation:          {:.3f} [{:5.2f}]'.format(time_preparation, 1 / time_preparation))
    print('iii) Load model (PyTorch): {:.3f} [{:5.2f}]'.format(time_model_pth, 1 / time_model_pth))
    print('iv)  Load model (TRT):     {:.3f} [{:5.2f}]'.format(time_model_trt, 1 / time_model_trt))
    print('v)   Computation:          {:.3f} [{:5.2f}]'.format(time_computation, 1/ time_computation))

    print('Network forward: PyTorch: {:.2e} [{:7.2f}], TRT: {:.2e} [{:7.2f}]'.format(avgmeter_pth.avg_time, avgmeter_pth.avg_fps, avgmeter_trt.avg_time, avgmeter_trt.avg_fps))
    print('GPU Synch.:      PyTorch: {:.2e} [{:7.2f}], TRT: {:.2e} [{:7.2f}]'.format(avgmeter_gpusynch_pth.avg_time, avgmeter_gpusynch_pth.avg_fps, avgmeter_gpusynch_trt.avg_time, avgmeter_gpusynch_trt.avg_fps))
    print('Error: {:.3e}'.format(statistics.mean(avgmeter_error.list_max)))
