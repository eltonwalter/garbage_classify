
# coding: utf-8
# In[1]:
#导入包
import os
import sys
import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
from nets import vgg
import glob

slim = tf.contrib.slim

ckpt=sys.argv[1]

# GPU设置
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


#参数设置
num_classes = 40
INPUT_IMAGE_SIZE = (224, 224)

img_path = '../data/valid/'
VGG_MEAN_rgb = [123.68, 116.779, 103.939]


#数据加载
def read_img(folder_dir):
    imgs = []
    labels = []
    for cls in os.listdir(folder_dir):
        for img_name in os.listdir(folder_dir+cls):
            image = Image.open(folder_dir+cls+'/'+img_name)
            img = image.resize((224,224),Image.ANTIALIAS)
            imgs.append(img)
            labels.append(int(cls))
    print(len(imgs),len(labels))

    return imgs, labels

#数据读取
def read_data(path):
    data, label = read_img(path)
    num_example = len(data)
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    rdata = []
    rlabel = []
    for i in arr:
        rdata.append(data[i])
        rlabel.append(label[i])
        
    s = np.int(num_example)
    x_train = rdata[:s]
    y_train = rlabel[:s]
    return x_train,y_train



x_train,y_train = read_data(img_path)


#模型初始化
img_input = tf.placeholder(tf.float32, shape=(224, 224, 3))
img_mean = img_input - VGG_MEAN_rgb
image_4d = tf.expand_dims(img_mean, 0)

reuse = True if 'vgg_model' in locals() else None

vgg_model = vgg.vgg_16
logits, _ = vgg_model(image_4d, num_classes, is_training=False, dropout_keep_prob=0, reuse=reuse)

# 计算每类概率
prediction = slim.softmax(logits)
prediction = tf.squeeze(prediction)

# 初始化计算图
isess.run(tf.global_variables_initializer())

#ckpt_filename = 'log_back/vgg/1/model.ckpt-3765'
ckpt_filename = 'logs/model.ckpt-'+str(ckpt)
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

predictions=[] 
num = 0
starttime=datetime.datetime.now()
for image in x_train:
    num += 1
    pred = isess.run([prediction],feed_dict={img_input:image})
    predictions.append(np.argmax(pred,axis=1)[0])
    print('共%s张，已预测%s张'%(len(x_train),num))
endtime=datetime.datetime.now()
print("totaltime:%s"%(endtime-starttime))


# In[12]:


def precision(predictions, targets):
    num_evals = len(predictions)
    c = 0
    j = 0
    k = 0
    for i in range(num_evals):
        if predictions[i] == targets[i]:
            c += 1
            if targets[i]==0:
                j += 1
            else:
                k += 1
    print("Precision:%s"%(c/len(predictions)))
        

precision(predictions, y_train)
print(predictions)
print(y_train)
