from mxnet.gluon import nn

'''
ConvBlock
[Conv2d ─ BN ─ ACTIVATION]
'''
def ConvBlock(channels, kernel_size, strides, padding, use_bias=False, activation='leaky'):
    block = nn.HybridSequential()
    block.add(nn.Conv2D(int(channels), kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias))

    if not use_bias:
        block.add(nn.BatchNorm(in_channels=int(channels)))

    if activation == 'leaky':
        block.add(nn.LeakyReLU(0.1))

    return block


'''
RedidualBlock
[   ┌───────────────────────┐    ]
[In ┴ Conv(1x1) ─ Conv(3x3) ┴ out]
'''
class ResidualBlock(nn.HybridBlock):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, kernel_size=1, strides=1, padding=0)
        self.conv2 = ConvBlock(channels, kernel_size=3, strides=1, padding=1)

    def hybrid_forward(self, F, x):
        # print('F: ', F)
        # print('x: ', x.shape, type(x))

        block = self.conv1(x)
        block = self.conv2(block)
        out = block + x

        return out


class CSP(nn.HybridBlock):
    def __init__(self, channels, block_size=1):
        super(CSP, self).__init__()
        self.conv0 = ConvBlock(channels, kernel_size=3, strides=2, padding=1)
        self.conv1 = ConvBlock(channels / 2, kernel_size=1, strides=1, padding=0)
        self.resblocks = self.make_residual_blocks(channels / 2, block_size)

        self.conv2 = ConvBlock(channels / 2, kernel_size=1, strides=1, padding=0)
        self.conv3 = ConvBlock(channels / 2, kernel_size=1, strides=1, padding=0)
        self.conv4 = ConvBlock(channels, kernel_size=1, strides=1, padding=0)

    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        short_cut = x
        x = self.conv1(x)
        x = self.resblocks(x)
        x = self.conv2(x)
        short_cut = self.conv3(short_cut)
        x = F.concat(x, short_cut, dim=1)
        x = self.conv4(x)

        return x

    def make_residual_blocks(self, channels, block_size):
        layer = nn.HybridSequential()
        for i in range(block_size):
            layer.add(ResidualBlock(channels))
        return layer


class DarkNet(nn.HybridBlock):
    def __init__(self, num_classes=1000, input_size=416):
        super(DarkNet, self).__init__()
        self.layer_num = 0
        self.num_classes = num_classes
        self.input_size = input_size

        self.input_layer = nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1, use_bias=False)

        self.layer1 = CSP(64, 4)
        self.layer2 = CSP(128, 4)
        self.layer3 = CSP(256, 8)
        self.layer4 = CSP(512, 8)
        self.layer5 = CSP(1024, 4)

        self.global_avg_pool = nn.GlobalAvgPool2D()
        self.fc = nn.Dense(self.num_classes)

    def hybrid_forward(self, F, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)

        return x


'''
from mxnet import nd
if __name__ == '__main__':
    daknet = DarkNet(num_classes=5, input_size=416)
    daknet.initialize()
    x = nd.random.normal(shape=(1, 3, 416, 416))
    y = daknet(x)
    print('y: ', y.shape, type(y))
'''
