import numpy as np
import torch
from torch import nn
from transformers import BeitFeatureExtractor, BeitForImageClassification
import math


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
        Input: (N,Cin,Hin,Win)
        Output: (N,Cout,Hout,Wout)
        Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    '''
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 8, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 4, kernel_size=4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 4, kernel_size=8, stride=1),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=8, stride=2),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 1, kernel_size=16, stride=1),
            self.make_gen_block(hidden_dim * 1, hidden_dim * 1, kernel_size=8, stride=2),
            self.make_gen_block(hidden_dim * 1, im_chan, kernel_size=37, stride=1, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
            Input: (N,Cin,Hin,Win)
            Output: (N,Cout,Hout,Wout)
            Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        z_dim of noise is considered as number of channels of input which each channel has size of 1*1
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        x1 = self.gen(x)
        return x1


class Critic(nn.Module):
    """
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, im_chan=3, hidden_dim=8):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim * 16, kernel_size=37, stride=1),
            self.make_crit_block(hidden_dim * 16, hidden_dim * 8, kernel_size=16, stride=2),
            self.make_crit_block(hidden_dim * 8, hidden_dim * 4, kernel_size=32, stride=1),
            self.make_crit_block(hidden_dim * 4, hidden_dim * 2, kernel_size=16, stride=2),
            self.make_crit_block(hidden_dim * 2, hidden_dim * 2, kernel_size=9, stride=2),
            self.make_crit_block(hidden_dim * 2, 1, kernel_size=7, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


class Generator1(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
        Input: (N,Cin,Hin,Win)
        Output: (N,Cout,Hout,Wout)
        Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    """
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):
        super(Generator1, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 8, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 4, kernel_size=4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 4, kernel_size=8, stride=1),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=8, stride=2),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 1, kernel_size=16, stride=1),
            self.make_gen_block(hidden_dim * 1, hidden_dim * 1, kernel_size=8, stride=2),
            self.make_gen_block(hidden_dim * 1, im_chan, kernel_size=37, stride=1, final_layer=True),
        )
        self.feature_extractor_BEIT = nn.Sequential(*list(BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k').children())[0:-1])
        self.preprocess = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
            Input: (N,Cin,Hin,Win)
            Output: (N,Cout,Hout,Wout)
            Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        z_dim of noise is considered as number of channels of input which each channel has size of 1*1
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        x1 = self.gen(x)
        x2 = self.feature_extractor_BEIT(x1)['pooler_output']
        feature = x2 / torch.linalg.norm(x2)
        return x1, feature


class unet(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
        Input: (N,Cin,Hin,Win)
        Output: (N, C_out, H_out, W_out)
        H_out=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    """
    def __init__(self, z_dim):
        super(unet, self).__init__()
        self.z_dim = z_dim
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=3, init_features=32, pretrained=False)
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(1, 1, kernel_size=8, stride=2),
            self.model
        )

    @staticmethod
    def make_gen_block(input_channels, output_channels, kernel_size=3, stride=1, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
            Input: (N,Cin,Hin,Win)
            Output: (N,C-out,H-out,Wout)
            H-out=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        z_dim of noise is considered as number of channels of input which each channel has size of 1*1
        """
        height = int(np.sqrt(self.z_dim))
        x = noise.view(len(noise), 1, height, height)
        x1 = self.gen(x)
        return x1


class japaness_generator(nn.Module):
    """
    Args:
        - num_voxel : int
        - noise_shape : tuple
    Inputs:
        - x : Tensor : (N, num_voxel)
    Outputs:
        - : Tensor : (N, 2)
    H-out = (Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    """
    def __init__(self, num_voxel, noise_shape=(256, 4, 4)):
        super(japaness_generator, self).__init__()

        self.noise_shape = noise_shape
        self.gen_1 = nn.Sequential(
            japaness_generator._block_FC(num_voxel, math.prod(noise_shape)),
            japaness_generator._block_FC(math.prod(noise_shape), math.prod(noise_shape)),
            japaness_generator._block_FC(math.prod(noise_shape), math.prod(noise_shape))
        )

        self.gen_2 = nn.Sequential(
            japaness_generator._block_UpConv2D(noise_shape[0], noise_shape[0], 4, 2),
            japaness_generator._block_UpConv2D(noise_shape[0], 2*noise_shape[0], 3, 1),
            japaness_generator._block_UpConv2D(2*noise_shape[0], noise_shape[0], 4, 2),
            japaness_generator._block_UpConv2D(noise_shape[0], noise_shape[0], 3, 1),
            japaness_generator._block_UpConv2D(noise_shape[0], int(noise_shape[0]/2), 4, 2),
            japaness_generator._block_UpConv2D(int(noise_shape[0]/2), int(noise_shape[0]/2), 3, 1),
            japaness_generator._block_UpConv2D(int(noise_shape[0]/2), int(noise_shape[0]/4), 4, 2),
            japaness_generator._block_UpConv2D(int(noise_shape[0]/4), int(noise_shape[0]/8), 4, 2),
            japaness_generator._block_UpConv2D(int(noise_shape[0]/8), 3, 4, 2)
        )

    def forward(self, x):
        return self.gen_2(
            self.gen_1(x).view((-1,)+self.noise_shape)
        )

    @staticmethod
    def _block_UpConv2D(in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.LeakyReLU(negative_slope=0.3)
        )

    @staticmethod
    def _block_FC(in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(negative_slope=0.3)
        )


class japaness_discriminator(nn.Module):
    """
    Args:
        - init_features : int
    Inputs:
        - x : Tensor : (N, C, H, W)
    Outputs:
        - : Tensor : (N, 2)
    """
    def __init__(self, init_features=32):
        super(japaness_discriminator, self).__init__()

        self.disc_1 = nn.Sequential(
            japaness_discriminator._block_Conv2D(3, init_features, 7, 4),
            japaness_discriminator._block_Conv2D(init_features, 2 * init_features, 5, 1),
            japaness_discriminator._block_Conv2D(2 * init_features, 4 * init_features, 3, 2),
            japaness_discriminator._block_Conv2D(4 * init_features, 8 * init_features, 3, 1),
            japaness_discriminator._block_Conv2D(8 * init_features, 8 * init_features, 3, 2),
            nn.AvgPool2d(11, 11)
        )

        self.disc_2 = nn.Sequential(
            japaness_discriminator._block_FC(8 * init_features, 8 * init_features),
            nn.ReLU(),
            japaness_discriminator._block_FC(8 * init_features, 1)
        )

    def forward(self, x):
        return self.disc_2(
            self.disc_1(x).flatten(start_dim=1)
        )

    @staticmethod
    def _block_Conv2D(in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.ReLU()
        )

    @staticmethod
    def _block_FC(in_features, out_features):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, out_features)
        )

