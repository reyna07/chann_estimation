
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




# Load the PyTorch model
pytorch_model = torch.load("cnn_model.pth")
golden_output_file = './estChannelGridPerfect.txt'
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
#parser.add_argument('--model_path', default="cnn_model.pth", help="Path to the trained PyTorch model")
parser.add_argument('--batch_size', default=1, type=int, help="Batch size for evaluation")
parser.add_argument('--quant_mode', default='test', choices=['float', 'calib', 'test'], help="Quantization mode")
parser.add_argument('--config_file', default=None, help='quantization configuration file')
parser.add_argument('--deploy', dest='deploy', action='store_true', help='export xmodel for deployment')
parser.add_argument('--inspect', dest='inspect',action='store_true',help='inspect model')
#parser.add_argument('--target', default="DPUCVDX8H_ISA1_F2W4_6PE_aieDWC", help="Target configuration for quantization")
#parser.add_argument('--data_file', default="data.mat", help="Path to the .mat file containing the validation data")
args = parser.parse_args()

# Target configuration for quantization
target = "DPUCVDX8H_ISA1_F2W4_6PE_aieDWC"
# target = "DPUCV2DX8G_ISA1_C20B104"
#quant_mode = 'test'
#quant_mode = 'calib'
#batch_size = 1
#batch_size = 32

quant_mode = args.quant_mode
batch_size = args.batch_size
config_file = args.config_file
deploy = args.deploy
inspect = args.inspect


def evaluate(model, val_loader, golden_output_file):
  delimiter='\t'
  model.eval()
  model = model.to(device)

  # Load the golden output from the text file manually.
  with open(golden_output_file, 'r') as file:
      golden_output = []
      for line in file:
          # Split the line into values using the specified delimiter and convert to floats.
          golden_output.extend(complex(value.replace('i','j')) for value in line.strip().split(delimiter))

  # Convert the golden output list to a PyTorch tensor.
  golden_tensor = torch.tensor(golden_output, dtype=torch.cfloat)
  predicted_flat = []
  golden_flat = golden_tensor.flatten()
    # Iterate through the DataLoader

      # Generate predictions
  outputs = model(val_loader)

      # Flatten predictions and add to the predicted list
  predicted_tensor = outputs.flatten()

  # Convert predicted_flat to a tensor

  # Calculate Mean Squared Error
  mse = ((golden_flat - predicted_tensor) ** 2).mean().item()

  print("Mean Squared Error (MSE) is being calculated...\n")
  return mse


if quant_mode != 'test' and deploy:
  deploy = False
  print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
if deploy and (batch_size != 1 ): #or subset_len != 1):
  print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
  batch_size = 1
  #subset_len = 1


# Generate random input tensor with the correct shape
input = torch.randn([batch_size, 1, 14, 612])

if quant_mode == 'float':
  quant_model = pytorch_model
  if inspect:
    if not target:
        raise RuntimeError("A target should be specified for inspector.")
    import sys
    from pytorch_nndct.apis import Inspector
    # create inspector
    inspector = Inspector(target)  # by name
    # start to inspect
    inspector.inspect(quant_model, (input,), device=device)
    sys.exit()

else:
  # Initialize the quantizer
  quantizer = torch_quantizer(
      quant_mode, pytorch_model, (input,), device=device, target=target
  )
  print("Quantization initialization is done!")
  # Get the quantized model
  quant_model = quantizer.quant_model

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Perform a forward pass to invoke the forward function
print("Performing a forward pass...\n")

output = quant_model(input)
if quant_mode == 'test':
  loss_final = evaluate(quant_model, input, golden_output_file)
  print('Accuracy as Mean Squared Error (MSE): %g' % (abs(loss_final)))
  print("\n")
##
#quantizer.load_ft_param()

  # to get loss value after evaluation
loss_fn = torch.nn.CrossEntropyLoss().to(device)
# For demonstration, create dummy target labels
#dummy_target = torch.randint(0, 10, (batch_size,), device=device)  # Assuming 10 classes

# Calculate loss
#loss = loss_fn(output, dummy_target)
#print(f"Loss: {loss.item()}")

# Export the quantized model
if quant_mode == 'calib':
  print("Exporting configuration file...\n")
  quantizer.export_quant_config()
  print("Config export complete!\n")


if deploy:
  print("Exporting models...\n")
  quantizer.export_torch_script()
  quantizer.export_onnx_model()
  quantizer.export_xmodel(deploy_check=True)

  print("Model export completed!")