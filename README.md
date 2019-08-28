# garbage_classify
通过tensorflow-slim实现垃圾分类网络的训练和预测，并生成包含参数的.pb模型。

其中train.py为训练文件，inference.py为测试文件，frozen.py将.ckpt参数文件转化为.pb模型。

getfiles.py的代码中，getfile1和getfile2函数分别对应不同的数据集格式。
get_files1: 图片按类别存放在不同的文件夹，文件夹名为类别名。
get_files2：文件夹中直接包含所有图像，每个图像有对应的.txt文件。
.txt文件中存放图像文件名及对应的标签，以逗号 , 隔开。
