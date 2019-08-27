# -*- coding: utf-8 -*-
import os
from nets import vgg
import tensorflow as tf
slim = tf.contrib.slim

# 参数设置
num_classes = 40
INPUT_IMAGE_SIZE = (224, 224)
VGG_MEAN_rgb = [123.68, 116.779, 103.939]

# GPU设置
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


def model_fn():
    reuse = True if 'vggnet' in locals() else None
    vggnet = vgg.vgg_16
    img_input = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    img_mean = img_input - VGG_MEAN_rgb
    logits, _ = vggnet(img_mean, num_classes, is_training=False, dropout_keep_prob=0.5, reuse=reuse)
    # 计算每类概率
    predictions = slim.softmax(logits)
    return img_input, predictions


def load_weights(weighs_file_path, pb_save_dir_local):

    inputs_, outputs_ = model_fn()
    isess.run(tf.global_variables_initializer())
    print('load weights from %s' % weighs_file_path)
    # 初始化计算图
    saver = tf.train.Saver()
    saver.restore(isess, weighs_file_path)
    print('load weights success')

    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_img': inputs_}, outputs={'output_score': outputs_})
    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(pb_save_dir_local, 'model'))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=isess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    print('save pb to local path success')


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    reuse = True if 'vggnet' in locals() else None
    vggnet = vgg.vgg_16
    img_input = tf.placeholder(tf.float32, shape=(None, None, 3))
    img_mean = img_input - VGG_MEAN_rgb
    image_4d = tf.expand_dims(img_mean, 0)
    logits, _ = vggnet(image_4d, num_classes, is_training=False, dropout_keep_prob=0.5, reuse=reuse)
    # 计算每类概率
    predictions = slim.softmax(logits)
    predictions = tf.squeeze(predictions)

    isess.run(tf.global_variables_initializer())
    print('load weights from %s' % input_checkpoint)

    # 初始化计算图
    saver = tf.train.Saver()
    saver.restore(isess, input_checkpoint)
    print('load weights success')

    output_node_names = "Squeeze"
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    tensor_name_list = [tensor.name for tensor in input_graph_def.node]
    for tensor_name in tensor_name_list:
        print(tensor_name, '\n')

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(os.path.join(output_graph, 'model.pb'), "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__=='__main__':
    input_checkpoint = 'logs/model.ckpt-1718'
    output_graph = 'models/'
    load_weights(input_checkpoint, output_graph)
    #freeze_graph(input_checkpoint, output_graph)