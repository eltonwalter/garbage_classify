
# coding: utf-8

# 导入包
import os
import tensorflow as tf
from glob import glob
from deployment import model_deploy
from nets import vgg

slim = tf.contrib.slim

# 配置信息：数据集位置及模型存储位置
folder_dir = '../train_data/'
output_path = 'logs/'

# 数据信息（类型数及数据总量）
num_classes = 40
num_samples_per_epoch = 14802

# 超参数:调参使用
batch_size = 64
init_learning_rate = 0.001
learning_rate_decay_factor = 0.96
num_epochs_per_decay = 0.5
decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
weight_decay = 0.00001
max_steps = 10000

# 服务器显卡数
num_clones = 2

# 其他初始化
VGG_MEAN_rgb = [123.68, 116.779, 103.939]


# 读取数据
def get_files(folder_dir):
    label_files = glob(os.path.join(folder_dir, '*.txt'))
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with open(file_path, 'r') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(folder_dir, img_name))
        labels.append(label)
    return img_paths,labels


# 读取batch
def reader():
    all_images_path, all_labels = get_files(folder_dir)
    file_dir_queue = tf.train.slice_input_producer([all_images_path, all_labels],shuffle=True,capacity=512)
    img_contents = tf.read_file(file_dir_queue[0])
    label = tf.cast(tf.one_hot(file_dir_queue[1], num_classes), tf.float32)
    image = tf.image.decode_jpeg(img_contents, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.3)
    image = image - VGG_MEAN_rgb
    image = tf.image.resize_images(image, [224,224], tf.image.ResizeMethod.BILINEAR, False)
    image_batch, label_batch = tf.train.batch([image, label], batch_size, capacity=128)
    return image_batch, label_batch


# 模型
def models(inputs, is_training=True, dropout_keep_prob=0.5):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_decay)):
        net, endpoints = vgg.vgg_16(inputs, num_classes, is_training=is_training, dropout_keep_prob=dropout_keep_prob)
    return net

def main(_):
    # 打印级别设置为info
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # 信息配置
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=num_clones,
            clone_on_cpu=False,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)
        # 全局步数
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()
        # 读取数据
        with tf.device(deploy_config.inputs_device()):
            # start = time.time()
            images, labels = reader()
            batch_queue = slim.prefetch_queue.prefetch_queue(
              [images, labels], capacity=2 * deploy_config.num_clones)
            # readtime = time.time()-start

        # 前向传播
        def clone_fn(batch_queue):
            images, labels = batch_queue.dequeue()
            logits = models(images)
            slim.losses.softmax_cross_entropy(logits, labels)
            return logits

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # 前向传播，每次取出一个batch进行计算
        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        # 取出目前计算到的梯度列表，保存在update_ops中
        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        with tf.device(deploy_config.optimizer_device()): 
            # 改变学习率，设置优化器
            # learning_rate = tf.train.piecewise_constant(global_step, [200, 500], [0.003, 0.0003, 0.0001])
            # staircase=True:每个poch更新学习率  False:每一步更新学习率

            learning_rate = tf.train.exponential_decay(init_learning_rate,
                                              global_step,
                                              decay_steps,
                                              learning_rate_decay_factor,
                                              staircase = False,
                                              name = 'exponential_decay_learning_rate')
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # 得到总损失，并计算梯度梯度，minimize()的第一部分
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=tf.trainable_variables())
        summaries.add(tf.summary.scalar('total_loss', total_loss))
        # 将计算出的梯度应用到变量上，minimize()的第二部分
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        # 保存update_ops
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        # 在执行with包含的内容前，先执行control_dependencies参数中的内容。
        # 即先执行update_op，再打印total_loss
        # 通过tf.identity创建节点，而不是直接运算
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')
            # tf.logging.info('readtime: %d' % readtime)

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,first_clone_scope))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # 指定需要加载的变量，加载预训练模型
        # fine_tune_path = 'checkpoints/vgg_16.ckpt'
        # variables_to_restore = slim.get_variables_to_restore(exclude=['global_step','vgg_16/fc8'])
        # init_fn = slim.assign_from_checkpoint_fn(fine_tune_path, variables_to_restore, ignore_missing_vars=True)
        # 配置GPU
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.9

        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        slim.learning.train(
            train_tensor,
            logdir=output_path,
            master='',
            is_chief=True,
            # init_fn=init_fn,
            init_fn=None,
            summary_op=summary_op,
            number_of_steps=max_steps,
            log_every_n_steps=1,
            save_summaries_secs=10,
            saver=saver,
            save_interval_secs=150,
            sync_optimizer=None,
            session_config=session_config)


if __name__ == '__main__':
    tf.app.run()

