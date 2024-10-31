import torch
import scipy.io
import numpy as np



def predict_and_convert_to_complex(model_path, input_mat_file):
    #The converted NN model - compatible with pytorch now
    channel_estimation_cnn = torch.load(model_path)
    channel_estimation_cnn.eval()

    mat = scipy.io.loadmat(input_mat_file)
    print(mat.keys())
    input_data = mat['nnInput']
    input_tensor = torch.from_numpy(input_data).float()
    #input_tensor = torch.from_numpy(input_data).to(torch.cfloat)
    print(input_tensor.size())
    input_tensor_transposed = np.transpose(input_tensor, (3, 2, 1, 0))
    print(input_tensor_transposed.size())
    with torch.no_grad():
        est_channel_grid_nn = channel_estimation_cnn(input_tensor_transposed)
    print(est_channel_grid_nn.size())
    # Convert results to complex numbers
    complex_est_channel_grid_nn = est_channel_grid_nn[0, :, :, :] + 1j * est_channel_grid_nn[1, :, :, :]
    print(complex_est_channel_grid_nn.size())
    return complex_est_channel_grid_nn




