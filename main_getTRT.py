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

