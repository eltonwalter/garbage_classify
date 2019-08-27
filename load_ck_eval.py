
# coding: utf-8
# In[1]:
#导入包
import os
import sys
import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#from nets import resnet_v2
from nets import cracknet
#from nets import alexnet
#from nets import vgg
#from nets import inception_v4
#from nets import mobilenet_v2

slim = tf.contrib.slim

import glob


# In[2]:
ckpt=sys.argv[1]

# GPU设置
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# In[3]:


#参数设置
num_classes = 2
INPUT_IMAGE_SIZE = (224, 224)
#SHOW_IMAGE_SIZE = (10, 10)

img_path = 'datasets/test/'
VGG_MEAN_rgb = [123.68, 116.779, 103.939]


# In[4]:


#数据加载
def read_img(path):
    folders = [path + x for x in os.listdir(path) if os.path.isdir(path)]
    imgs   = []
    labels = []
    num = [0,0]
    i = 0
    for idx, folder in enumerate(folders):
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the image: %s' % (im))
            num[i] += 1
            image = Image.open(im)
            img = image.resize((224,224),Image.ANTIALIAS)
            imgs.append(img)
            labels.append(idx)
        print(num[i])
        i += 1
    return imgs, labels


# In[5]:


#数据读取
def read_data(path):
    data, label = read_img(path)
    num_example = len(data)
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    #print(arr)
    rdata = []
    rlabel = []
    for i in arr:
        rdata.append(data[i])
        rlabel.append(label[i])
        
    s = np.int(num_example)
    x_train = rdata[:s]
    y_train = rlabel[:s]
    x_val   = rdata[s:]
    y_val   = rlabel[s:]
    return x_train,y_train


# In[6]:



x_train,y_train = read_data(img_path)


# In[7]:


#模型初始化
img_input = tf.placeholder(tf.float32, shape=(224, 224, 3))
img_mean = img_input - VGG_MEAN_rgb
image_4d = tf.expand_dims(img_mean, 0)
#image_4d_grey = tf.image.rgb_to_grayscale(image_4d)

reuse = True if 'vgg_model' in locals() else None
vgg_model = cracknet.cracknet
#vgg_model = resnet_v2.resnet_v2_101
#vgg_model = alexnet.alexnet_v2
#vgg_model = vgg.vgg_16
#vgg_model = mobilenet_v2.mobilenet_v2
#vgg_model = inception_v4.inception_v4
logits, _endpoints = vgg_model(image_4d, num_classes, is_training=False)
#logits, _endpoints = vgg_model(image_4d_grey, num_classes, is_training=False)
#logits, _endpoints = vgg_model(image_4d, num_classes, is_training=False, dropout_keep_prob=0)
#logits, _endpoints = vgg_model(image_4d, num_classes, is_training=False, dropout_keep_prob=0, reuse=reuse)
#计算每类概率
prediction = slim.softmax(logits)
#predictions = tf.squeeze(predictions)

#初始化计算图
isess.run(tf.global_variables_initializer())

#ckpt_filename = 'log_back/vgg/1/model.ckpt-3765'
ckpt_filename = 'logs/model.ckpt-'+str(ckpt)  #2591#3024
#ckpt_filename = 'logs/model.ckpt-79404'
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)


# In[11]:


predictions=[] 
num = 0
starttime=datetime.datetime.now()
for image in x_train:
    num += 1
    pred = isess.run([prediction],feed_dict={img_input:image})[0].tolist()[0]
    #print(pred[0])
    if pred[0]>0.5:
        predictions.append(0)
    else:
        predictions.append(1)  
    if num%50 ==0:
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
        if int(int(predictions[i])) == targets[i]:
            c += 1
            if targets[i]==0:
                j += 1
            else:
                k += 1
    #print("class1:%s,class2:%s",(j,k))
    print("Precision:%s"%(c/len(predictions)))
    #print(num)
    print("Precision1:%s"%(2*j/len(predictions)))
    print("Precision2:%s"%(2*k/len(predictions)))
        
        


# In[13]:



#print(predictions)
#print(y_train)
precision(predictions, y_train)

