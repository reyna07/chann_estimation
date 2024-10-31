# EECE 7398 FPGAs in the Cloud Project: Channel Estimation

## Introduction

## Setting up the project
1. Follow the [Getting started with Alveo V70s in OCT](https://github.com/OCT-FPGA/versal-tutorials/blob/main/v70-getting-started.md) to set the environment in Open Cloud Testbed.

   
2. Setup the Environment Variable after enter the Docker container.

```
source /workspace/board_setup/v70/setup.sh DPUCV2DX8G_v70
```

3. Clone this github to Vitis-AI directory

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

Then, you can see the result:

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
