
# coding: utf-8

'''

本文件功能为导入数据集。
输入为文件夹路径，输出为所有图片路径列表img_paths及其对应的标签列表labels。
get_file1: 图片按类别存放在不同的文件夹。
get_files2：文件夹中直接包含所有图像，每个图像有对应的.txt文件。
.txt文件中存放图像文件名及对应的标签，以逗号 , 隔开。
'''
import os
from glob import glob

def get_files1(folder_dir):
    img_paths = []
    labels = []
    for cls in os.listdir(folder_dir):
        for img_name in os.listdir(folder_dir + cls):
            img_path = os.path.join(folder_dir + cls + '/' + img_name)
            img_paths.append(img_path)
            labels.append(int(cls))
    return img_paths,labels

def get_files2(folder_dir):
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