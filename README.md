## Deformable Convolution形变卷积模块 --Pytorch
---

## 目录  
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项) 
3. [效果展示 Effect](#效果展示)

## 所需环境  
1. Python3.7
2. Pytorch>=1.10.1+cu113  
3. Torchvision>=0.11.2+cu113
4. Numpy==1.19.5
5. Pillow==8.2.0
6. Opencv-contrib-python==4.5.1.48
7. CUDA 11.0+
8. Cudnn 8.0.4+

## 注意事项  
1. 一定程度上解决Pytorch的tensor坐标切片无法并行映射特征的缺点(无gather_nd)  
2．Pytorch的grid_sample()操作类似于Opencv的remap()，无法对特征通道进行映射，通过拆分特征通道实现其映射
3. 坐标切片勿使用min-max归一化，特征损失严重

## 效果展示  
Tensorflow形变卷积  
![image](https://github.com/JJASMINE22/Deformable-Convolution-Pytorch/blob/main/sample/tensorflow.jpg)  
Pytorch形变卷积  
![image](https://github.com/JJASMINE22/Deformable-Convolution-Pytorch/blob/main/sample/torch.jpg)  

