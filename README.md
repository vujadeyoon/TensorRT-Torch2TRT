# TensorRT-Torch2TRT


## Table of contents
1.  [Notice](#notice)
2.  [Summarized environments about the TensorRT-Torch2TRT](#envs)
3.  [How to install the TensorRT](#tensorrt)
4.  [How to install the Torch2TRT](#torch2trt)
5.  [Save and Load model for the Torch2TRT](#torch2trt_save_load)
6.  [Run a demo code to compare the results between the PyTorch and the TensorRT using the Torch2TRT](#demo)
7.  [Results](#results)


## 1. Notice <a name="notice"></a>
- A guide for TensorRT and Torch2TRT
    - The TensorRT does not support any virtual envrionments such as virtualenv and conda. <br />
      In other words, the TensorRT only supports root-environment or docker. <br />
      In this guide, I describe the TensorRT on root-environment, not docker.
- Both TensorRT and Torch2TRT are officially researched and developed by the NVIDIA.
- The word, TRT that I mention in this README.md means TensorRT.
- The Torch2TRT only support single-batch process, not multi-batch process.
- This guide includes how to install the Torch2TRT with plugins. 
- I recommend that you should ignore the commented instructions with an octothorpe, #.
- Modified date: Sep. 13, 2020.


## 2. Summarized environments about the TensorRT-TRT <a name="envs"></a>
- Operating System (OS): Ubuntu MATE 18.04.3 LTS (Bionic)
- Graphics Processing Unit (GPU): NVIDIA TITAN Xp, 1ea
- GPU driver: Nvidia-440.100
- CUDA toolkit: CUDA 10.2
- cuDNN: cuDNN v7.6.5
- Python3: Python 3.6.9
- PyCUDA: 2019.1.2
- PyTorch: 1.6.0
- Torchvision: 0.7.0
- TensorRT: 7.0.0.11
- Torch2TRT: 0.1.0 (with plugins)


## 3. How to install the TensorRT <a name="tensorrt"></a>
A. Reference to the website,
<a href="https://developer.nvidia.com/tensorrt" title="TensorRT">TensorRT</a>.<br />

B. Download the TensorRT suitable for the development environment.<br />
- Tar File Install Packages For Linux x86
    - TensorRT 7.0.0.11 for Ubuntu 18.04 and CUDA 10.2 tar package (i.e. TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz)<br />
    
C. Preparations.
```bash
usrname@hostname:~/curr_path$ mkdir -p /home/usrname/pip3_packages
usrname@hostname:~/curr_path$ cp -r TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz /home/usrname/pip3_packages/
usrname@hostname:~/curr_path$ cd /home/usrname/pip3_packages/TensorRT-7.0.0.11
```

D. Uncompress the downloaded tar.gz file.
- Please note that you do not remove the uncompressed directory, TensorRT-7.0.0.11.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ tar -xzvf TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz
```

E. Register an environmental variable.
- Check the current CUDA toolkit environmental variable, LD_LIBRARY_PATH.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ $LD_LIBRARY_PATH
```
```bash
    bash: /usr/local/cuda-10.2/lib64: Is a directory
```
- Register a TensorRT path to the environmental variable, LD_LIBRARY_PATH.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/usrname/pip3_packages/TensorRT-7.0.0.11/lib
```
- The above things should be done every time when the shell script is initially executed. <br />
I do not recommend it, but if you want to permanently register environment variables, you can run the command below. 
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ echo -e "\n## TensorRT paths"  >> ~/.bashrc
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/usrname/pip3_packages/TensorRT-7.0.0.11/lib' >> ~/.bashrc
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ source ~/.bashrc
```

F. Copy plugins of the TensorRT to the CUDA toolkit directory.
- I recommend you should backup the original CUDA toolkit directory, /usr/local/cuda-10.2/targets.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo mkdir -p  /usr/local/cuda-originals/cuda-10.2
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo cp -r /usr/local/cuda-10.2/targets/ /usr/local/cuda-originals/cuda-10.2/
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo cp -r /home/usrname/pip3_packages/TensorRT-7.0.0.11/include/* /usr/local/cuda-10.2/targets/x86_64-linux/include/
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo cp -r /home/usrname/pip3_packages/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/* /usr/local/cuda-10.2/targets/x86_64-linux/lib/
```

G. Install python packages.
- PyCUDA >= 2019.1.1
- PyTorch >= 1.3.0
- Check the default python3 version on the root-envirionment.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ python3 --version
```
```bash
    Python 3.6.9
```
- Installation
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ pip3 install "pycuda>=2019.1.1"
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ pip3 install torch torchvision
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo pip3 install ./python/tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl
```
H. Install UFF packages.
- Installation
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo pip3 install ./uff/uff-0.6.5-py2.py3-none-any.whl
```
- Check the UFF path.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ which convert-to-uff
```
```bash
    /usr/local/bin/convert-to-uff
```
I. Install graphsurgeon packages.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo pip3 install ./graphsurgeon/graphsurgeon-0.4.1-py2.py3-none-any.whl
```


## 4. How to install the Torch2TRT <a name="torch2trt"></a>
A. Preparations.
```bash
usrname@hostname:~/curr_path$ mkdir -p /home/usrname/pip3_packages
usrname@hostname:~/curr_path$ cd /home/usrname/pip3_packages
usrname@hostname:~/pip3_packages$ 
```
B. Clone the git from NVIDIA-AI-IOT/torch2trt.
- I recommend you should not remove the cloned directory, torch2trt.
```bash
usrname@hostname:~/pip3_packages$ git clone https://github.com/NVIDIA-AI-IOT/torch2trt
usrname@hostname:~/pip3_packages$ cd torch2trt
usrname@hostname:~/pip3_packages/torch2trt$
```
C. Install the Torch2TRT.<br />
- Option 1: Without plugins
```bash
usrname@hostname:~/pip3_packages/torch2trt$ sudo python3 setup.py install
```
- Option 2: With plugins
    - When you fail to install the Torch2TRT with plugins and the below error is observed, the plugins of the TensorRT may not be copied correctly.
  Please refer to the Section 3-F in order to copy the plugins of the TensorRT correctly.
    ```bash
    torch2trt/plugins/interpolate.cpp:6:10: fatal error: NvInfer.h: No such file or directory
     #include <NvInfer.h>
              ^~~~~~~~~~~
    compilation terminated.
    error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
    ```
```bash
usrname@hostname:~/pip3_packages/torch2trt$ sudo python3 setup.py install --plugins
```


## 5. Save and Load model for the Torch2TRT <a name="torch2trt_save_load"></a>
A. Example.
```python
import torch
from torch2trt import torch2trt, TRTModule

# Save a TRT model from the PyTorch model
x = torch.rand((args.batch, args.channel, args.height, args.width)).to(device)
model_trt_1 = torch2trt(model_pytorch, [x])
torch.save(model_trt_1.state_dict(), args.path_ckpt_trt)
 
# Load the saved TRT model
model_trt_2 = TRTModule()
model_trt_2.load_state_dict(torch.load(args.path_ckpt_trt))
```


## 6. Run a demo code to compare the results between the PyTorch and the TensorRT using the Torch2TRT <a name="demo"></a>
A. Clone the git.
```bash
usrname@hostname:~/curr_path$ git clone https://github.com/vujadeyoon/TensorRT-Torch2TRT
usrname@hostname:~/curr_path$ cd TensorRT-Torch2TRT
usrname@hostname:~/curr_path/TensorRT-Torch2TRT$
```
B. Register a TensorRT path to the environmental variable, LD_LIBRARY_PATH.
```bash
usrname@hostname:~/curr_path/TensorRT-Torch2TRT$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/usrname/pip3_packages/TensorRT-7.0.0.11/lib
```
- For reference, unregister the temporal registered environmental variable, LD_LIBRARY_PATH.
```bash
usrname@hostname:~/curr_path/TensorRT-Torch2TRT$ unset LD_LIBRARY_PATH 
```
C. Download pretrained weights from the PyTorch Hub.
- Example model: ResNet-18
```bash
usrname@hostname:~/curr_path/TensorRT-Torch2TRT$ wget https://download.pytorch.org/models/resnet18-5c106cde.pth
```
D. Get the TensorRT model using the Torch2TRT from the PyTorch model for a customized ResNet-18.
- To check the function of the Torch2TRT plugins, the interpolation function was added into the ResNet-18 model.
- Python script: resnet.py
```python
import torch.nn.functional as F

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        return self._forward_impl(x)
```
- Python script: main_getTRT.py
```python
import time
import argparse
import torch
from resnet import resnet18
from torch2trt import torch2trt


parser = argparse.ArgumentParser(description='A example for TensorRT and Torch2TRT')
parser.add_argument('--path_ckpt_pth', type=str, default='./resnet18-5c106cde.pth', help='Path for a pretrained PyTorch weights.')
parser.add_argument('--path_ckpt_trt', type=str, default='.', help='Path for a TRT weights.')
parser.add_argument('--batch', type=int, default=1, help='A batch of an input tensor')
parser.add_argument('--channel', type=int, default=3, help='A channel of an input tensor')
parser.add_argument('--height', type=int, default=224, help='A height of an input tensor')
parser.add_argument('--width', type=int, default=224, help='A width of an input tensor')
args = parser.parse_args()


if __name__ == '__main__':
    if args.batch != 1:
        raise ValueError('The args.batch should be 1 because the Torch2TRT does not support multi-batch size.')

    time_preparation_tic = time.time()
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    time_start = time.time()

    model_pth = resnet18().eval().to(device)
    model_pth.load_state_dict(torch.load(args.path_ckpt_pth))
    x = torch.rand((args.batch, args.channel, args.height, args.width)).to(device)
    model_trt = torch2trt(model_pth, [x])

    torch.save(model_trt.state_dict(), './resnet18_{}x{}-trt.pth'.format(args.height, args.width))

    time_end = time.time()

    print('Time [sec.]: {:.3f}'.format(time_end - time_start))
```
```bash
usrname@hostname:~/curr_path/TensorRT-Torch2TRT$ python3 main_getTRT.py
Time [sec.]: 8.129
```
E. Compare results between PyTorch and the TensorRT using the Torch2TRT for the ResNet-18.
- Python script: main_run.py
```python
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

    print('Network forward: PyTorch: {:.2e} [{:.2e}], TRT: {:.2e} [{:.2e}]'.format(avgmeter_pth.avg_time, avgmeter_pth.avg_fps, avgmeter_trt.avg_time, avgmeter_trt.avg_fps))
    print('GPU Synch.:      PyTorch: {:.2e} [{:.2e}], TRT: {:.2e} [{:.2e}]'.format(avgmeter_gpusynch_pth.avg_time, avgmeter_gpusynch_pth.avg_fps, avgmeter_gpusynch_trt.avg_time, avgmeter_gpusynch_trt.avg_fps))
    print('Error: {:.3e}'.format(statistics.mean(avgmeter_error.list_max)))
```
```bash
# The error means averaged maximum absolute difference.

usrname@hostname:~/curr_path/TensorRT-Torch2TRT$ python3 main_compare.py
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 192.86it/s]
Time profile. unit: sec. [FPS]
i)   Total:                3.432 [ 0.29]
ii)  Preparation:          0.017 [60.33]
iii) Load model (PyTorch): 1.578 [ 0.63]
iv)  Load model (TRT):     1.277 [ 0.78]
v)   Computation:          0.560 [ 1.78]
Network forward: PyTorch: 2.20e-03 [ 455.08], TRT: 1.73e-04 [5781.47]
GPU Synch.:      PyTorch: 6.80e-04 [1469.87], TRT: 1.73e-03 [ 576.62]
Error: 3.399e-06
```

## 7. Results <a name="results"></a>
- Unit: sec. [FPS]
- Inference speed (Network forward): The averaged inference speed of the TensorRT using the Torch2TRT is much faster than that of the PyTorch without any other hardware-acceleration.
    - PyTorch:  2.20e-03 [ 455.08]
    - TensorRT: 1.73e-04 [5781.47]
- GPU latency (GPU synchronization): The averaged gpu synchronization speed of the TensorRT using the Torch2TRT is faster than that of the PyTorch without any other hardware-acceleration.
    - PyTorch:  6.80e-04 [1469.87]
    - TensorRT: 1.73e-03 [ 576.62]
- Accuracy: There is no difference in accuracy between the PyTorch and TensorRT using Torch2TRT computations.
    - Error: 3.399e-06
    - Error is calculated between the PyTorch and the TensorRT using the Torch2TRT by the averaged maximum absolute difference.
    
