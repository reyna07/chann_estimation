import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision

pytorch_model = torch.load("cnn_model.pth")
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = "DPUCV2DX8G_ISA1_C20B104"
quant_mode = 'calib'
batch_size = 32
input = torch.randn([batch_size, 1, 14, 612])


quantizer = torch_quantizer(
    quant_mode, pytorch_model, (input), device=device, target=target)
print("quantization is done!")
    # Get the converted model to be quantized.
quant_model = quantizer.quant_model


  # to get loss value after evaluation
loss_fn = torch.nn.CrossEntropyLoss().to(device)


quantizer.export_torch_script()
quantizer.export_onnx_model()
quantizer.export_xmodel()
