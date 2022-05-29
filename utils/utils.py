from functools import reduce

import numpy as np
from PIL import Image


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    #        三个通道                    # 灰度图
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
'''参数：
    image：输入图片
    size：图片的大小（416*416）
    letterbox_images:是否为小图片（比416小的）
'''
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size            # 获得图片的宽和高
    w, h    = size                  # 获得需要转变为的尺寸
    if letterbox_image:
        scale   = min(w/iw, h/ih)   # 获得原图和需要转变图片宽高最小的那个比例
        nw      = int(iw*scale)     # 尝试恢复到原来尺寸（但会有偏差）
        nh      = int(ih*scale)     # 尝试恢复到原来尺寸（但会有偏差）

        image   = image.resize((nw,nh), Image.BICUBIC) # BICUBIC插值法，改变图片的形状
        new_image = Image.new('RGB', size, (128,128,128)) # 填充图片
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))    # 随机裁剪图片
    else: # 如果图片大于远需要的图片，直接改变大小，不用填充
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
''':arg
    输入图片种类的地址，通过read函数读取，即可获得种类以及个数
    返回种类以及个数
'''
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
''':arg
    同理，跟上面的获得类一样，最终返回先验框以及个数
'''
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)
