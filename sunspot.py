# 数据读取
import numpy as np
import paddle.fluid as fluid
import os
from PIL import Image

def read_input():
    def read_data():
        file_path = "./enhance_img/"
        classes  = ["alpha", "beta", "betax"]
        for cls_id, cls in enumerate(classes ):
            imgage_list = os.listdir(file_path + cls)
            for image_name in imgage_list:
                img_size = 256
                img = Image.open(file_path + cls + "/" + image_name).convert("RGB")
                # print(img.shape)
                # x = input()
                img.show()
                img = img.resize((img_size, img_size), Image.ANTIALIAS)
                img = np.array(img).reshape(1, 3, img_size, img_size).astype(np.float32)
                # lab = [0, 0, 0]
                lab = cls_id
                yield img, lab
    return read_data

# 训练函数
import os
import random
import paddle
import paddle.fluid as fluid
import numpy as np

def train(model):
    print('start read data ... ')
    model.train()
    epoch = 10
    batch_size = 8
    # opt = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameter_list=model.parameters())
    # opt = fluid.optimizer.SGD(learning_rate=0.01)
    # opt = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
    opt = fluid.optimizer.AdamOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    print("start training ...")
    for i in range(epoch):
        reader = fluid.io.shuffle(read_input(), buf_size=14469)
        train_reader = fluid.io.batch(reader, batch_size=batch_size)
        for batch_id, data in enumerate(train_reader()):
            # optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            x_data = np.array([item[0] for item in data], dtype='float32').reshape((-1, 3, 256, 256))
            # print(x_data.shape)
            y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            logits = model(img)
            # print(label)
            # print(logits)
            # 计算损失函数
            loss = fluid.layers.softmax_with_cross_entropy(logits, label)
            avg_loss = fluid.layers.mean(loss)
            if batch_id % 100 == 0:
                print("epoch:{}, batch_id: {}, loss is: {}".format(i, batch_id, avg_loss.numpy()))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()

# ResNet模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        """
        
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)

# 定义ResNet模型
class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=3):
        """
        
        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            #ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]
        
        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1, # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        # 创建全连接层，输出大小为类别数目
        self.out = Linear(input_dim=2048, output_dim=class_dim,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

        
    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        return y
"""开始训练"""
use_gpu = True #使用GPU的方法
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

with fluid.dygraph.guard():
    model = ResNet()
    train(model)
fluid.save_dygraph(model.state_dict(), 'sloar_spot_ResNet')



"""================必须先执行前面的训练函数，最后执行 test 函数================"""
"""开始测试"""
# 读取测试集
from PIL import Image
import os
import numpy as np
def read_data_input():
    file_path = "./test_img/"
    image_list = os.listdir(file_path)
    images = []
    name = []
    for path in image_list:
        img_size = 256
        img = Image.open(file_path + path).convert('RGB')
        # print(img.shape)
        img.show()
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        img = np.array(img).reshape(1, 3, img_size, img_size).astype(np.float32)
        images.append(img)
        name.append(path)
    return images, name
# 将测试集的结果保存在 txt 中
def save_to_txt(list):
    data = list
    f = open("./test_resnet_1.txt", "w")
    for i in range(len(data)):
        f.write(str(data[i][0]).strip("'") + str(" ") + str(data[i][1]) + str("\n"))
        # f.write(str(data[i]) + str("\n"))
        # print(str(data[i][0]).strip("'") + str(" ") + str(data[i][1]))
    f.close()

def  test():
    import numpy
    data, name= read_data_input()
    with fluid.dygraph.guard():
        print('start evaluation .......')
        model = ResNet()
        #加载模型参数
        model_state_dict, _ = fluid.load_dygraph("sloar_spot_ResNet.pdparams")
        model.load_dict(model_state_dict)
        model.eval()
        results = []
        for i in range(len(data)):
            img = fluid.dygraph.to_variable(data[i])
            out = model(img)
            out = out.numpy().reshape(1, -1)
            # print(out)
            for j in range(out.shape[0]):
                result = []
                out = out[j,:].tolist()
                out = int(out.index(max(out))) + int(1)
                result.append(name[i])
                result.append(out)
                results.append(result)
        save_to_txt(results)