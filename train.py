from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.yolo import get_train_model, yolo_body
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.dataloader import YoloDatasets
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

# 获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    # 设置当前程序可见的设备范围
    tf.config.experimental.set_memory_growth(gpu, True)

#----------------------------------------------------#
if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用eager模式训练
    #----------------------------------------------------#
    eager = False
    #--------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #--------------------------------------------------------#
    classes_path    = 'model_data/my_classes.txt'
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #-------------------------------------------------------------------------------------#
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #   预训练权重对于99%的情况都必须要用，不用的话权值太过随机，特征提取效果不明显
    #   网络训练的结果也不会好，数据的预训练权重对不同数据集是通用的，因为特征是通用的
    #------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolo_weights.h5'
    #------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape     = [416, 416]
    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch          = 0          # 默认epoch
    Freeze_Epoch        = 50         # 冻结的epoch数
    Freeze_batch_size   = 8          # 冻结网络批数据量的大小
    Freeze_lr           = 1e-3       # 学习率
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100        # 到多少epoch解冻网络
    Unfreeze_batch_size = 2          # 解冻网络批数据量的大小
    Unfreeze_lr         = 1e-4       # 学习率
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，0代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   keras里开启多线程有些时候速度反而慢了许多
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------#
    num_workers         = 4
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    # 返回种类以及数据个数
    class_names, num_classes = get_classes(classes_path)
    # 返回检验框以及个数
    anchors, num_anchors     = get_anchors(anchors_path)

    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    model_body  = yolo_body((None, None, 3), anchors_mask, num_classes)
    #------------------------------------------------------#
    #   载入预训练权重：加快神经网络的收敛
    #------------------------------------------------------#
    print('Load weights {}.'.format(model_path))
    model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    if not eager:
        model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)
    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址，可输入：tensorboard --logdir= 目录，进行网页可视化
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式：这里使用指数衰减算法
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.92, verbose = 1)
    # EarlyStopping 提前停止防止过拟合。monitor：监控的数量；min_delta：小于该值的会被当成模型没有进步
    #                               patience：没有进步的训练轮数，在这之后训练就会被停止；verbose：详细信息模式
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    loss_history    = LossHistory('logs/') # 保存训练的过程

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f: # 训练集
        train_lines = f.readlines()
    with open(val_annotation_path) as f: # 验证集
        val_lines   = f.readlines()
    num_train   = len(train_lines)      # 计算训练集数量
    num_val     = len(val_lines)        # 计算验证集数量

    if Freeze_Train:
        freeze_layers = 184 # 解冻184层
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        epoch_step      = num_train // Unfreeze_batch_size    # 一个epoch的训练个次数
        epoch_step_val  = num_val   // Unfreeze_batch_size

        # 如果数据为0，则提醒用户
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Freeze_batch_size))

        if eager:
            # 加载训练集
            gen     = tf.data.Dataset.from_generator(partial(
                YoloDatasets(train_lines, input_shape, anchors, Freeze_batch_size, num_classes, anchors_mask, train = True).generate
            ), (tf.float32, tf.float32, tf.float32, tf.float32))
            # 加载验证集
            gen_val = tf.data.Dataset.from_generator(partial(
                YoloDatasets(train_lines, input_shape, anchors, Freeze_batch_size, num_classes, anchors_mask, train = False).generate
            ), (tf.float32, tf.float32, tf.float32, tf.float32))
            # 打乱数据集
            gen     = gen.shuffle(buffer_size = Freeze_batch_size).prefetch(buffer_size = Freeze_batch_size)
            gen_val = gen_val.shuffle(buffer_size = Freeze_batch_size).prefetch(buffer_size = Freeze_batch_size)
            # 学习率指数衰减策略
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = Freeze_lr, decay_steps = epoch_step, decay_rate=0.92, staircase=True)
            # 加载优化器
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
            # 训练
            for epoch in range(Init_Epoch, Freeze_Epoch):
                fit_one_epoch(model_body, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            Freeze_Epoch, input_shape, anchors, anchors_mask, num_classes)

        else:
            # 加载优化器
            model.compile(optimizer=Adam(lr = Freeze_lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            # 加载数据集
            train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, Freeze_batch_size, num_classes, anchors_mask, train = True)
            val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, Freeze_batch_size, num_classes, anchors_mask, train = False)
            # 训练模型
            model.fit_generator(
                generator           = train_dataloader,  # 批量训练数据集
                steps_per_epoch     = epoch_step,        # 训练的次数
                validation_data     = val_dataloader,    # 批量训练验证
                validation_steps    = epoch_step_val,    # 验证集的步长
                epochs              = Freeze_Epoch,      # 训练的epoch个数
                initial_epoch       = Init_Epoch,        # 初始的poch数
                use_multiprocessing = True if num_workers != 0 else False,
                workers             = num_workers, # 使用多少线程工作
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]    )# 各个类的继承

    if Freeze_Train: # 解冻所有已冻结的网络
        for i in range(freeze_layers): model_body.layers[i].trainable = True


    if True:
        epoch_step      = num_train // Unfreeze_batch_size # 一个epoch的训练个次数
        epoch_step_val  = num_val   // Unfreeze_batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Unfreeze_batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(
                YoloDatasets(train_lines, input_shape, anchors, Unfreeze_batch_size, num_classes, anchors_mask, train = True).generate
            ), (tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(
                YoloDatasets(train_lines, input_shape, anchors, Unfreeze_batch_size, num_classes, anchors_mask, train = False).generate
            ), (tf.float32, tf.float32, tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = Unfreeze_batch_size).prefetch(buffer_size = Unfreeze_batch_size)
            gen_val = gen_val.shuffle(buffer_size = Unfreeze_batch_size).prefetch(buffer_size = Unfreeze_batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = Unfreeze_lr, decay_steps = epoch_step, decay_rate=0.92, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(Freeze_Epoch, UnFreeze_Epoch):
                fit_one_epoch(model_body, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            UnFreeze_Epoch, input_shape, anchors, anchors_mask, num_classes)

        else:
            model.compile(optimizer=Adam(lr = Unfreeze_lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

            train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, Unfreeze_batch_size, num_classes, anchors_mask, train = True)
            val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, Unfreeze_batch_size, num_classes, anchors_mask, train = False)

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = UnFreeze_Epoch,
                initial_epoch       = Freeze_Epoch,
                use_multiprocessing = True if num_workers != 0 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )
