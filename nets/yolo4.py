from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, LeakyReLU, \
    BatchNormalization,Input,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from nets.CSPdarknet53 import darknet_body
from utils.utils import compose

from nets.yolo_training import yolo_loss

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# --------------------------------------------------#
#   单次卷积
# --------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def make_five_convs(x, num_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def yolo_body(inputs_shape, num_anchors, num_classes):
    num_anchors=len(num_anchors)
    # 生成darknet53的主干模型
    inputs=Input(inputs_shape)
    feat1, feat2, feat3 = darknet_body(inputs)

    # 第一个特征层
    # y1=(batch_size,13,13,3,85)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    # 使用了SPP结构，即不同尺度的最大池化后堆叠。
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)

    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(P5)

    P4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    P4 = Concatenate()([P4, P5_upsample])
    P4 = make_five_convs(P4, 256)

    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(P4)

    P3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = make_five_convs(P3, 128)

    P3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P3)
    print(num_anchors)
    print((num_classes + 5))
    P3_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P3_output)

    # 38x38 output
    P3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(P3_downsample)
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_five_convs(P4, 256)

    P4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)

    # 19x19 output
    P4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(P4_downsample)
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_five_convs(P5, 512)

    P5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])



def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(3)]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask, 'num_classes' : num_classes, 'ignore_thresh': 0.5}
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
