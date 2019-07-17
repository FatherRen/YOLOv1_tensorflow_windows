# -- encoding:utf-8 --
"""
Created by MengCheng Ren on 2019/7/14
"""
import os


# 训练集路径
DATA_PATH = 'data'
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
# pkl数据文件存放路径
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
# 训练模型输出路径
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
# 预训练模型地址
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

# 预训练模型
WEIGHTS_FILE = 'data/weights/YOLO_small.ckpt'

# 训练的模型存储路径
MODEL = './model/yolo'
# summary存储路径
SUMMARY_PATH = './summary'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

TEST_DATA = './test_data'

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

LEARNING_RATE = 0.00001

DECAY_STEPS = 3000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 45

MAX_ITER = 15000

SUMMARY_ITER = 10

SAVE_ITER = 100


#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
