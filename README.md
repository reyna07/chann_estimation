# EECE 7398 FPGAs in the Cloud Project: Channel Estimation

## Introduction

## Setting up the project
1. Follow the [Getting started with Alveo V70s in OCT](https://github.com/OCT-FPGA/versal-tutorials/blob/main/v70-getting-started.md) to set the environment in Open Cloud Testbed.

   
2. Setup the Environment Variable after enter the Docker container.

```
$ source /workspace/board_setup/v70/setup.sh DPUCV2DX8G_v70
```

3. Clone this github to Vitis-AI directory

```
$ git clone https://github.com/reyna07/chann_estimation.git
```

## Software Simulation

Run the `run.sh` script to see the software simulation result:
```
$ chmod +x run.sh
$ ./run.sh
```

Then, you can see the result:
>dict_keys(['__header__', '__version__', '__globals__', 'nnInput'])
>torch.Size([612, 14, 1, 2])
>torch.Size([2, 1, 14, 612])
>torch.Size([2, 1, 14, 612])
>torch.Size([1, 14, 612])
>torch.Size([1, 14, 612])
>dict_keys(['__header__', '__version__', '__globals__', 'estChannelGridPerfect'])
>torch.Size([612, 14])
>torch.Size([14, 612])
>tensor([0.2351-0.1000j, 0.2383-0.1098j, 0.2260-0.1330j,  ...,
>        0.2679-0.1260j, 0.3576-0.1346j, 0.3738-0.1624j])
>tensor([0.6302-0.4570j, 0.6019-0.4140j, 0.5802-0.3642j,  ...,
>        0.5086-0.4028j, 0.4964-0.3710j, 0.4921-0.3317j])
>tensor([0.2836, 0.2247, 0.1789,  ..., 0.1346, 0.0752, 0.0426])
>Mean Squared Error between predicted and golden output: 0.0697585940361023
