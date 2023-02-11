import tensorflow as tf
from tqdm import tqdm
from nets.yolo import yolo_loss

# 防止bug
def get_train_step_fn(input_shape, anchors, anchors_mask, num_classes):
    @tf.function
    def train_step(imgs, targets, net, optimizer):
        with tf.GradientTape() as tape: # 梯度带
            # 计算loss
            # 输出三个预测框
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args        = [P5_output, P4_output, P3_output] + targets
            # 调用损失函数
            loss_value  = yolo_loss(args, input_shape, anchors, anchors_mask, num_classes)
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        # 对各参数求梯度
        grads = tape.gradient(loss_value, net.trainable_variables)
        # 梯度更新
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        # 返回损失
        return loss_value
    return train_step

# 训练
def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, 
            input_shape, anchors, anchors_mask, num_classes):
    ''':arg
        net：网络
        loss_history：保存训练的过程的地址
        optimizer：优化器
        epoch：迭代了第几次
        epoch_step：批训练迭代的次数
        epoch_step_val：验证集迭代的次数
        gen：训练集
        gen_val：验证集
        Epoch：迭代的终点
        input_shape：输入图片的尺寸
        anchors：先验框
        anchors_mask：用于帮助代码找到对应的先验框
        num_classes：检测的个数
    '''
    train_step  = get_train_step_fn(input_shape, anchors, anchors_mask, num_classes)
    loss        = 0 # 训练的损失
    val_loss    = 0 # 验证的损失
    print('Start Train')
    # Tqdm是一个快速，可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):# 迭代训练集
            if iteration >= epoch_step: # 如果迭代的次数达到批训练数时，退出
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2] # 三个先验框
            targets     = [tf.convert_to_tensor(target) for target in targets] # 三个先验框
            loss_value  = train_step(images, targets, net, optimizer) # 计算损失
            loss        = loss + loss_value # 损失叠加

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer._decayed_lr(tf.float32).numpy()}) # 输出
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val: # 如果迭代的次数达到批训练数时，退出
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2] # 三个先验框
            targets     = [tf.convert_to_tensor(target) for target in targets] # 三个先验框
            # 计算loss
            P5_output, P4_output, P3_output = net(images)
            args        = [P5_output, P4_output, P3_output] + targets
            loss_value  = yolo_loss(args, input_shape, anchors, anchors_mask, num_classes)
            loss_value  = tf.reduce_sum(net.losses) + loss_value
            val_loss = val_loss + loss_value # 损失叠加

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    # 输出这次迭代损失
    logs = {'loss': loss.numpy() / (epoch_step+1), 'val_loss': val_loss.numpy() / (epoch_step_val+1)}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    # 保存模型
    net.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % ((epoch + 1), loss / (epoch_step + 1) ,val_loss / (epoch_step_val + 1)))
