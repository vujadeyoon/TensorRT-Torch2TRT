# TensorRT-Torch2TRT
- A guide for TensorRT and Torch2TRT
    - The TensorRT does not support any virtual envrionments such as virtualenv and conda. <br />
      In other words, the TensorRT only supports root-environment or docker. <br />
      In this guide, I describe the TensorRT on root-environment, not docker.
- Both TensorRT and Torch2TRT are officially researched and developed by the NVIDIA.
- The word, TRT that I mention in this README.md means TensorRT. 
- I recommend that you should ignore the commented instructions with an octothorpe, #.
- Modified date: Aug. 23, 2020.


## Table of contents
1.  [Summarized environments about the TensorRT-Torch2TRT](#envs)
2.  [How to install the TensorRT](#tensorrt)
3.  [How to install the Torch2TRT](#torch2trt)
4.  [Save and Load model for the Torch2TRT](#torch2trt_save_load)
5.  [Run a demo code to compare the results between the PyTorch and the TensorRT using the Torch2TRT](#demo)
6.  [Results](#results)


## 1. Summarized environments about the TensorRT-TRT <a name="envs"></a>
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
- Torch2TRT: 0.1.0


## 2. How to install the TensorRT <a name="tensorrt"></a>
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
F. Install python packages.
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
G. Install UFF packages.
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
H. Install graphsurgeon packages.
```bash
usrname@hostname:~/pip3_packages/TensorRT-7.0.0.11$ sudo pip3 install ./graphsurgeon/graphsurgeon-0.4.1-py2.py3-none-any.whl
```


## 3. How to install the Torch2TRT <a name="torch2trt"></a>
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
```bash
usrname@hostname:~/pip3_packages/torch2trt$ sudo python3 setup.py install --plugins
```


## 4. Save and Load model for the Torch2TRT <a name="torch2trt_save_load"></a>
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


## 5. Run a demo code to compare the results between the PyTorch and the TensorRT using the Torch2TRT <a name="demo"></a>
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
D. Get the TensorRT model using the Torch2TRT from the PyTorch model for ResNet-18.
- Python script: main_getTRT.py
```python
import time
import argparse
import torch
from resnet import resnet18
from torch2trt import torch2trt


parser = argparse.ArgumentParser(description='A example for TensorRT and Torch2TRT')
parser.add_argument('--path_ckpt_pytorch', type=str, default='./resnet18-5c106cde.pth', help='Path for a pretrained PyTorch weights.')
parser.add_argument('--path_ckpt_trt', type=str, default='./resnet18-trt.pth', help='Path for a TRT weights.')
parser.add_argument('--batch', type=int, default=1, help='A batch of an input tensor')
parser.add_argument('--channel', type=int, default=3, help='A channel of an input tensor')
parser.add_argument('--height', type=int, default=224, help='A height of an input tensor')
parser.add_argument('--width', type=int, default=224, help='A width of an input tensor')
args = parser.parse_args()


if __name__ == '__main__':
    time_preparation_tic = time.time()
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    time_start = time.time()

    model_pytorch = resnet18().eval().to(device)
    model_pytorch.load_state_dict(torch.load(args.path_ckpt_pytorch))
    x = torch.rand((args.batch, args.channel, args.height, args.width)).to(device)
    model_trt = torch2trt(model_pytorch, [x])

    torch.save(model_trt.state_dict(), args.path_ckpt_trt)

    time_end = time.time()

    print('Time [sec.]: {:.3f}'.format(time_end - time_start)) # Time [sec.]: 7.719

```
```bash
usrname@hostname:~/curr_path/TensorRT-Torch2TRT$ python3 main_getTRT.py
Time [sec.]: 7.719
```
E. Compare results between PyTorch and the TensorRT using the Torch2TRT for the ResNet-18.
- Python script: main_run.py
```python
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

```
```bash
# The error means maximum absolute difference.

usrname@hostname:~/curr_path/TensorRT-Torch2TRT$ python3 main_compare.py
Time profile [sec.]
i)   Total:                3.393
ii)  Preparation:          0.019
iii) Load model (PyTorch): 1.842
iv)  Load model (TRT):     1.485
v)   Computation:          0.047
[0/9] Time: PyTorch: 4.094e-03, TRT: 4.008e-04 / Error: 5.126e-06
[1/9] Time: PyTorch: 2.885e-03, TRT: 2.041e-04 / Error: 4.530e-06
[2/9] Time: PyTorch: 2.232e-03, TRT: 1.900e-04 / Error: 4.053e-06
[3/9] Time: PyTorch: 2.207e-03, TRT: 1.533e-04 / Error: 5.722e-06
[4/9] Time: PyTorch: 2.524e-03, TRT: 1.898e-04 / Error: 2.861e-06
[5/9] Time: PyTorch: 2.263e-03, TRT: 1.552e-04 / Error: 3.576e-06
[6/9] Time: PyTorch: 2.284e-03, TRT: 1.404e-04 / Error: 2.503e-06
[7/9] Time: PyTorch: 2.286e-03, TRT: 1.485e-04 / Error: 2.444e-06
[8/9] Time: PyTorch: 2.265e-03, TRT: 1.454e-04 / Error: 4.649e-06
[9/9] Time: PyTorch: 2.283e-03, TRT: 1.481e-04 / Error: 3.457e-06
```

## 6. Results <a name="results"></a>
- Inference speed [sec.]: The averaged inference speed of the TensorRT using the Torch2TRT is much faster than that of the PyTorch without any other hardware-acceleration.
    - PyTorch:  2.532e-03
    - TensorRT: 1.876e-04
- Accuracy: There is no difference in accuracy between the PyTorch and TensorRT using Torch2TRT computations.
    - Error: 3.892e-06
    - Error is calculated between the PyTorch and the TensorRT using the Torch2TRT by the averaged maximum absolute difference.
    
