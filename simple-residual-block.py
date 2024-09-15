import torch
import torch.nn as nn

class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, activation=nn.ReLU):
        """
        A customizable residual block for reusability in PyTorch models.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernels (default is 3x3).
            stride: Stride of the convolution (default is 1).
            padding: Padding for the convolution (default is 1).
            activation: Activation function (default is ReLU).
        """
        super(SimpleResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation1 = activation()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation2 = activation()

    def forward(self, x):
        """
        Forward pass for the residual block.

        Args:
            x: Input tensor.
        
        Returns:
            Tensor: Output tensor after applying the residual block operations.
        """
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv2(out)
        return self.activation2(out)
