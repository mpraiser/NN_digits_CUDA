# 神经网络手写数字识别/CUDA加速

[TOC]

## 文件说明
1. `gpu-info.py`
    输出关于当前机器GPU的相关信息。

2. `neural_networks_digits_cpu.py`
    CPU实现。

3. `neural_networks_digits_gpu.py`
    使用GPU进行加速。依赖于`neural_networks_digits_cpu`。

## 依赖
1. NumPy
2. PyCUDA
3. CUDA
4. Visual Studio