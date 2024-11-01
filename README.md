# EECE 7398 FPGAs in the Cloud Project: Channel Estimation

Rana Bogrekci - Chunan Chen

## Introduction
Channel Estimation is a concept of wireless communication in which the process tries to figure out the channel characteristics. In all communication systems, data is transmitted from one place to another. The medium which a signal is transmitted through is called the channel. The channel can be both wired and wireless, and it can distort the signal while it passes. To be able to remove the distortion on the signal, we need to know what characteristics the channel carries, and in what way it distorts the signal.  Neural networks have become a popular scheme in channel estimation.
![The Process of Channel Estimation with Neural Networks](https://github.com/reyna07/chann_estimation/blob/main/img/ch_estimation_process.png)

For this project, we aim to imagine the channel as a 2D matrix having two axes depicting the time and frequency response, and turn the problem of channel estimation into an image processing problem, where 2D CNNs commonly used. To speed up the process of inference, we plan to use VCK5000. 

## Model and data acquisition
The Matlab example:[Deep Learning Data Synthesis for 5G Channel Estimation](https://www.mathworks.com/help/5g/ug/deep-learning-data-synthesis-for-5g-channel-estimation.html) provides us with a pre-trained channel evaluation model, input data for testing, and golden output data as a reference (perfect evaluation, actual channel realization).

## Project realization
We ported the channel evaluation model generated by the Matlab example to the Pytorch environment to realize the inference process. In the testbench, we use the test signals generated by the Matlab example as inputs of inference, and the output of inference are compared with the golden outputs (perfect evaluation, realization of the actual channel provided by the Matlab example). The testbench will return the MSE(Mean Square Error) of the inference output against the golden output.

Then, we will use the Vitis-AI Pytorch workflow to do the quantization and optimization to the model, finally implement the model to VCK5000.

## Setting up the project
1. Follow the [Getting started with VCK5000 Versal devices in OCT](https://github.com/OCT-FPGA/versal-tutorials/blob/main/vck5000-getting-started.md) to set the environment in Open Cloud Testbed.

   
2. Go to the Vitis-AI directory

```
cd /docker/Vitis-AI
```

3. Start the Docker container

```
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```

4. Setup the Environment Variable after enter the Docker container.

```
source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_8pe_normal 
```

5. Clone this github to Vitis-AI directory

```
git clone https://github.com/reyna07/chann_estimation.git
```

## Software Simulation
1. Get into the `chann_estimation` directory
```
cd chann_estimation
```

2. Run the `run.sh` script to see the software simulation result:
```
chmod +x run.sh
./run.sh
```

Then, you can see the result which shows the MES between SW simulation result and golden output:

```
Prediction of the pretrained model has started...

Prediction finished!

Prediction file is ready!

Comparing against output data:

*****************************************************************************

PASS: The output matches the golden output!

Mean Squared Error between predicted and golden output: 0.0697585865855217

*****************************************************************************

```
