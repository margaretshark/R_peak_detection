import torch
import torch.nn as nn

class Downsample(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.apply_batchnorm = apply_batchnorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)
        x = nn.functional.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_padding=0, apply_dropout=False):
        super(Upsample, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=output_padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.apply_dropout = apply_dropout

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batchnorm(x)
        x = nn.functional.relu(x)
        if self.apply_dropout:
            x = self.dropout(x)
        return x

class Conv1DTranspose(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, strides=2, activation=None):
        super(Conv1DTranspose, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            input_channel, output_channel, kernel_size, stride=strides, padding=(kernel_size-1)//2, output_padding=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class UNet_1D(nn.Module):
    def __init__(self):
        super(UNet_1D, self).__init__()
        self.down_stack = nn.ModuleList([
            Downsample(1, 16, 9, apply_batchnorm=False), # (bs, 7232, 1)
            Downsample(16, 16, 9),
            Downsample(16, 32, 6),
            Downsample(32, 32, 6), 
            Downsample(32, 64, 3), 
            Downsample(64, 64, 3), 
        ])
        self.up_stack = nn.ModuleList([
            Upsample(64, 64, 3, 1), # (bs, 226, 1)
            Upsample(128, 32, 3, 1), # (bs, 452, 1)
            Upsample(64, 32, 6), # (bs, 904, 1)
            Upsample(64, 16, 6), # (bs, 1808, 1)
            Upsample(32, 16, 9, 1), # (bs, 3616, 1)
        ])
        self.last = Conv1DTranspose(32, 1, 9, strides=2, activation=nn.Sigmoid())
        self.inputs = nn.Identity()

    def forward(self, x):
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
            
        skips = reversed(skips[:-1])
        
        for up, skip in zip(self.up_stack, skips):

            x = up(x)
            x = torch.cat([x, skip], dim=1) # B,C,L over channels             
            
        x = self.last(x)
        return x   