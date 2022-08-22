# Image-reconstruction-from-Brain-activity
Coco dataset would be used in order to train GAN dataset. URL related to this dataset are following:
http://images.cocodataset.org/zips/val2017.zip 

http://images.cocodataset.org/zips/test2017.zip

http://images.cocodataset.org/zips/unlabeled2017.zip

http://images.cocodataset.org/zips/train2017.zip

# Description of different versions:
## trainV1: 
In this script we use WGAN for generating data. Because of this kind of discriminator which is called
wasserstein loss, run time would be very high. Also, real images from coco dataset would be fed into discriminator.
In order to train generator there are 2 kinds of loss: 
1-MSE loss 
2-Discriminator loss
## trainV2:
It is same as trainV1. The only difference is adding comparator loss for training generator
and also each loss has its own special weight which changes the affect of each loss in training process.
## trainV3:
It is same as trainV1 with a little changes in loss which is eliminating gradient penalty
## trainV4:
It is same as trainV1 but a very simple generator in order to prevent mode collapse and also the input shape is changed from 11000 channels to single channel
## trainV5:
It is same as previous two version but a network is added in order to reconstruct fMRI from generated image.

In addition, there is two choice for normalization. first method is min-max and second is mean_std method
which the second one seems lead to better results
In next versions categorical cross entropy would be used which reduces runtime significantly
Up to now, it seems that increasing parameters of gradient loss of discriminator
will result in improvement in quality of outputs.

# probable problems in training process:
1- high number of model parameters
2- type of loss func including wloss and cross entropy loss
3- using of pretrained networks because of low number of data
4- maybe eliminating wloss would be result in best outcomes

# Ideas:
1- getting histogram of image from fmri signal
2- using UNet to create image from fmri and vice versa
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=3, init_features=32, pretrained=True)
3- changing fmri data into the shape of image


# Best results till now:
trainV1 which min-max normalization and gradient penalty with the 
parameter of 5 and batch size of 8 are used. also there are mse and wloss.
