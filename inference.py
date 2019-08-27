#导入包
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from nets import vgg
slim = tf.contrib.slim

# GPU设置
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

#参数设置
num_classes = 40
INPUT_IMAGE_SIZE = (224, 224)
VGG_MEAN_rgb = [123.68, 116.779, 103.939]

#模型初始化
reuse = True if 'vggnet' in locals() else None

ckpt_filename = 'logs/model.ckpt-747'

img_input = tf.placeholder(tf.float32, shape=(None, None, None, 3))
img_mean = img_input - VGG_MEAN_rgb
#image_4d = tf.expand_dims(img_mean, 0)
vggnet = vgg.vgg_16
logits, endpoints = vggnet(img_mean, num_classes, is_training=False, dropout_keep_prob=0.5, reuse=reuse)

#计算每类概率
predictions = slim.softmax(logits)

#初始化计算图
isess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

#推断
def inference(images):
    #获得predictions
    pred = isess.run([predictions], feed_dict={img_input: images})
    print(pred)
    #求概率最大类
    pred_cls = np.argmax(pred[0], axis=1)
    return pred_cls


TEST_IMAGE_PATHS = 'demo'
def main():
    total_img = os.listdir(TEST_IMAGE_PATHS)
    image_list = []
    if not total_img:
        print('NO images found')
        return
    for image_path in sorted(total_img):
    #打开图像并调整图像大小
        img = os.path.join(TEST_IMAGE_PATHS,image_path)
        image = Image.open(img)
        image = image.resize(INPUT_IMAGE_SIZE,Image.ANTIALIAS)
        (im_width, im_height) = image.size
        image = np.array(image.getdata()).reshape((im_width, im_height, 3)).astype(np.uint8)
        image_list.append(image)
    #推断并返回gradient
    cls = inference(image_list)
    print(cls)

if __name__ == '__main__':
    main()
