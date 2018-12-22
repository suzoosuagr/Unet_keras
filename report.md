# Datasets
## Membrane
Including 30 membrane images (24 for training, 6 for validation), all gray. 
## ISIC2018
Including 2332 skin images (2075 for training, 259 for validation), all rgb images.

# Architectures 
## Unet_Laplacian_v1
[]






# 部分总结
For ISIC2018 输入是rgb or gray 无所谓，不会对结构有太多的影响. 

# CODE log
+ @2018-12-17 在原本Unet的代码上做出的更改跑不出来，于是还是回归到github上的范例代码，将数据的访问结构调整到适合ISICkeras数据集的结构。

+ @2018-12-20 cGAN的时候 weights 的保存名字没有写清。 **For RGB weights name**: 'ISIC_gray_inputs_iou_bce_L_1_v3.hdf5' **For gray inputs weights**： 'ISIC_gray_truely_inputs_iou_bce_L_1_v3.hdf5'