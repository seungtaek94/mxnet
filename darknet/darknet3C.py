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
        self.conv1 = ConvBlock(channels / 2, kernel_size=1, strides=1, padding=0)
        self.conv2 = ConvBlock(channels, kernel_size=3, strides=1, padding=1)

    def hybrid_forward(self, F, x):
        block = self.conv1(x)
        block = self.conv2(block)
        out = block + x

        return out


class DarkNet(nn.HybridBlock):
    def __init__(self, num_classes=1000, input_size=416):
        super(DarkNet, self).__init__()
        self.layer_num = 0
        self.num_classes = num_classes
        self.input_size = input_size

        self.input_layer = nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1, use_bias=False)

        self.layer1 = self.make_layer(64, 1)
        self.layer2 = self.make_layer(128, 2)
        self.layer3 = self.make_layer(256, 8)
        self.layer4 = self.make_layer(512, 8)
        self.layer5 = self.make_layer(1024, 4)

        self.top_branch_layer = self.make_branch_layer()
        self.bottom_branch_layer = self.make_branch_layer()

        self.global_avg_pool = nn.GlobalAvgPool2D()
        self.fc = nn.Dense(self.num_classes)

    def hybrid_forward(self, F, x):
        x_top, x_bottom = F.split(x, axis=2, num_outputs=2)

        # DarkNet53
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Brach Layer
        x_top = self.top_branch_layer(x_top)
        x_bottom = self.bottom_branch_layer(x_bottom)

        # Concat branch with darknet
        x = F.concat(x_top, x, x_bottom, dim=1)

        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x

    def make_layer(self, channels, layer_size=1):
        layer = nn.HybridSequential()

        layer.add(ConvBlock(channels, kernel_size=3, strides=2, padding=1))

        for i in range(layer_size):
            layer.add(ResidualBlock(channels))

        return layer

    def make_branch_layer(self):
        layer = nn.HybridSequential()
        layer.add(ConvBlock(channels=32, kernel_size=7, strides=(1, 2), padding=3))
        layer.add(nn.MaxPool2D(pool_size=(3, 3), strides=2, padding=1))
        layer.add(ConvBlock(channels=64, kernel_size=3, strides=1, padding=1))
        layer.add(ConvBlock(channels=128, kernel_size=3, strides=2, padding=1))
        layer.add(ConvBlock(channels=256, kernel_size=3, strides=2, padding=1))
        layer.add(ConvBlock(channels=512, kernel_size=3, strides=2, padding=1))

        return layer

'''
if __name__ == '__main__':
    daknet = DarkNet(num_classes=5, input_size=416)
    daknet.initialize()
    x = nd.random.normal(shape=(1, 3, 416, 416))
    y = daknet(x)
    print('y: ', y.shape, type(y))
'''