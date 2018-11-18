# Datasets
## Membrane
Including 30 membrane images (24 for training, 6 for validation), all gray. 
## ISIC2018
Including 2332 skin images (2075 for training, 259 for validation), all rgb images.

# Experiment 1
## Parameters
- **MODEL**: Using Unet with `binary cross entropy` as loss, `Adam` with learning rate 1e-3. 
- **DATA** : Membrane datasets, batch size 16, epoch: 20, with augmentation. 
## Result:
The results shown as 






# 部分总结
For ISIC2018 输入是rgb or gray 无所谓，不会对结构有太多的影响. 
