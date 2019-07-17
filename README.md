# YOLOv1_tensorflow_windows
# 使用起来简洁明了，将各个模块封装成类方法，方便大家学习使用。
## 1、介绍：
    此版本YOLOv1只包含6个文件：
      data：存放训练数据，预训练模型，处理后的训练数据pkl文件。
      model：存储自己迭代训练后保存的模型。
      summary：存放日志文件，用于可视化。
      test_data：在测试时所用到的图像或者视频文件。
      config.py：包含有模型中的一些参数的设置。
      main.py：包含有数据的加载，YOLOv1模型的构建和训练，以及在训练好模型后进行测试的各个模块。
      
## 2、使用：
    （1）要求：python3.6、tensorflow-gpu——1.8.0、cv2——3.4.2
    （2）数据集以及预训练模型：我以上传到[百度云](https://pan.baidu.com/s/1-xMmsHbkigPZfl4jQ4JtKw）提取码：hm8n，
         下载完数据后将数据解压后直接替换掉原文件中的data文件夹即可。
    （3）准备工作完毕：打开cmd，cd到本文件夹。
         1、训练模型：虽然下载下来的数据中有预训练模型，但是还是需要进行进行训练后才能进行测试。这是代码里面默认图设置的时候的原因。
         如果你不更改训练的数据就不需要训练多久，只要model文件夹中生成了ckpt文件就可以了。
         python main.py --is_training True
         2、测试模型：当模型训练完毕之后运行下面这行命令，文件名是你存放在目录中test_data文件夹中的图像或者视频文件的文件名；
         data_type的类型必须是image或者video，如果要测试的是图像就设置image，如果是视频就选择video。
         python main.py --is_training False --test_data '文件名(可以时图像或者视频)' --data_type '可选image或者video'

## 3、注意：
    如果要训练自己的数据，在GitHub上找一个VOC数据集制作工具制作标签，存放在对应的路径下既可。（打标签是一个繁琐的工作...）
