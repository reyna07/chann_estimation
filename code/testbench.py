import torch
from chann_estimator import predict_and_convert_to_complex
import scipy.io
from sklearn.metrics import mean_squared_error
import numpy as np

model_path = './cnn_model.pth'
input_file = 'nnInput.mat'
golden_output_file = 'goldenOutput'
prediction = predict_and_convert_to_complex(model_path, input_file)
print(prediction.size())
golden_data = scipy.io.loadmat(golden_output_file)
print(golden_data.keys())
golden_output = golden_data['estChannelGridPerfect']
golden_tensor = torch.from_numpy(golden_output).to(torch.cfloat)
print(golden_tensor.size())
golden_output_transposed = np.transpose(golden_tensor, ( 1, 0))
print(golden_output_transposed.size())
predicted_flat = prediction.flatten()
golden_flat = golden_output_transposed.flatten()
print(predicted_flat)
print(golden_flat)

ert = pow(np.abs(golden_flat - predicted_flat), 2)
print(ert)
# Calculate Mean Squared Error (MSE)
mse = np.mean(pow(np.abs(golden_flat - predicted_flat), 2).numpy())
print(f"Mean Squared Error between predicted and golden output: {mse}")






