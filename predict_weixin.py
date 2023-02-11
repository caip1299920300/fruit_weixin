#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from yolo import YOLO
import base64



gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    yolo = YOLO()
    import os.path
    from flask import Flask
    from flask import app, request

    app = Flask(__name__)  # 创建1个Flask实例
    basedir = os.path.abspath(os.path.dirname(__file__))  # 定义一个根目录 用于保存图片用

    label = ''
    image = []
    urled = ''

    @app.route('/getimage', methods=['POST'])
    def getimage():
        global label
        label = ''
        global image
        global urled
        # 获取图片文件 name = file
        img = request.files.get('file')
        # 定义一个图片存放的位置 存放在static下面
        path = basedir + "/static/img/"
        # 图片名称
        imgName = img.filename
        # 图片path和名称组成图片的保存路径
        file_path = path + imgName
        # 保存图片
        img.save(file_path)
        # url是图片的路径
        url = '/static/img/' + imgName
        # image = Image.open('./img/street.jpg')
        image = Image.open(file_path)
        print(image)

        r_image, label = yolo.detect_image(image)
        print('========:', url)
        urled = url + ';' + "/static/imged/" + imgName
        r_image.save(basedir + "/static/imged/" + imgName)
        return urled


    if __name__ == "__main__":
        # print(app.url_map)
        # app.run()
        app.run(host='0.0.0.0', port=5000)