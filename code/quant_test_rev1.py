import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torchvision.transforms as transforms
import torch
import torchvision

# Load the PyTorch model
pytorch_model = torch.load("cnn_model.pth")
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/code/"
# Target configuration for quantization
target = "DPUCVDX8H_ISA1_F2W4_6PE_aieDWC"
# target = "DPUCV2DX8G_ISA1_C20B104"
#quant_mode = 'test'
quant_mode = 'calib'
batch_size = 32

# Generate random input tensor with the correct shape
input = torch.randn([batch_size, 1, 14, 612])

# Initialize the quantizer
quantizer = torch_quantizer(
    quant_mode, pytorch_model, (input,), device=device, target=target)

print("Quantization initialization is done!")

# Get the quantized model
quant_model = quantizer.quant_model

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss().to(device)



if quant_mode == 'calib':
    
# Perform a forward pass to invoke the forward function
  print("Performing a forward pass...")
  output = quant_model(input)

  quantizer.export_quant_config() 


  # to get loss value after evaluation
loss_fn = torch.nn.CrossEntropyLoss().to(device)
# For demonstration, create dummy target labels
#dummy_target = torch.randint(0, 10, (batch_size,), device=device)  # Assuming 10 classes

# Calculate loss
#loss = loss_fn(output, dummy_target)
#print(f"Loss: {loss.item()}")
if quant_mode == 'test':
  # Export the quantized model
  print("Exporting models...")

  quantizer.export_torch_script()
  quantizer.export_onnx_model()
  quantizer.export_xmodel()

  print("Model export completed!")
