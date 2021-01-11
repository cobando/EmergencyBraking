import torch
import torch.nn as nn

class SpatialFiltering(nn.Module):
    def __init__(self, n_channels, n_virtual_channels):
        super(SpatialFiltering, self).__init__()

        self.transform = nn.Conv2d(in_channels=1, out_channels=n_virtual_channels, kernel_size=(n_channels, 1),
                                   bias=False)

        print('>> SpatialFiltering has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # return self.transform(x).permute(0, 2, 3, 1)
        return self.transform(x).permute(0, 2, 1, 3)
        # return self.transform(x).permute(0, 2, 1) # For modes


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, convolution_size, pool_size):
        super(ConvBlock, self).__init__()

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1, convolution_size), padding=(0, convolution_size // 2)),
            # nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
        )
        self.activation = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        )

        print('>> ConvBlock has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        return self.activation(self.transform(x))


class DenseBlock(nn.Module):
    def __init__(self, in_features):
        super(DenseBlock, self).__init__()

        self.transform = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),   # p probability of an element to be zeroed.
            nn.Linear(in_features, 1),
            # nn.Softmax(dim=1),  #
            # nn.Sigmoid(),  # Comment if using BCEWithLogitsLoss()
        )

        print('>> DenseBlock has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        return self.transform(x.view(x.size(0), -1))


class DreemNet(nn.Module):
    def __init__(self, n_channels, n_virtual_channels, convolution_size, pool_size, n_hidden_channels,
                 window_size):
        super(DreemNet, self).__init__()

        self.spatial_filtering = SpatialFiltering(n_channels, n_virtual_channels)
        self.conv_block_1 = ConvBlock(in_channels=1, out_channels=n_hidden_channels, # in_channels=1, images would be 3
                                      convolution_size=convolution_size, pool_size=pool_size)
        self.conv_block_2 = ConvBlock(in_channels=n_hidden_channels, out_channels=n_hidden_channels,
                                      convolution_size=convolution_size, pool_size=pool_size)
        self.dense_block = DenseBlock(in_features=(window_size // pool_size ** 2 * n_virtual_channels * n_hidden_channels))



        print('>> DreemNet has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.spatial_filtering(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.dense_block(x)
        return x


