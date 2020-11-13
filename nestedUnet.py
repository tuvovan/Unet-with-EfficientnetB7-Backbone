from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,UpSampling2D,MaxPooling2D,Concatenate


class conv_block_nested(Model):

    def __init__(self, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = Activation('relu')
        self.conv1 = Conv2D(mid_ch, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(out_ch, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class NestedUNet(Model):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, classes=16):
        super(NestedUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = MaxPooling2D(strides=2)
        self.Up = UpSampling2D()

        self.conv0_0 = conv_block_nested(filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0], filters[0])

        self.final = Conv2D(classes, kernel_size=1,activation='softmax',name='final_layer')

    def call(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(Concatenate()([x0_0, self.Up(x1_0)]))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(Concatenate()([x1_0, self.Up(x2_0)]))
        x0_2 = self.conv0_2(Concatenate()([x0_0, x0_1, self.Up(x1_1)]))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(Concatenate()([x2_0, self.Up(x3_0)]))
        x1_2 = self.conv1_2(Concatenate()([x1_0, x1_1, self.Up(x2_1)]))
        x0_3 = self.conv0_3(Concatenate()([x0_0, x0_1, x0_2, self.Up(x1_2)]))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(Concatenate()([x3_0, self.Up(x4_0)]))
        x2_2 = self.conv2_2(Concatenate()([x2_0, x2_1, self.Up(x3_1)]))
        x1_3 = self.conv1_3(Concatenate()([x1_0, x1_1, x1_2, self.Up(x2_2)]))
        x0_4 = self.conv0_4(Concatenate()([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)]))

        output = self.final(x0_4)
        return output