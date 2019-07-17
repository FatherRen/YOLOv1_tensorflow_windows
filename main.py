# -- encoding:utf-8 --
"""
Created by MengCheng Ren on 2019/7/16
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import cv2
import pickle
import copy
import config as cfg
from tensorflow.contrib import slim
import argparse


class Data(object):
    def __init__(self):
        # 数据集的路径
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        # 保存数据集的pkl文件
        self.cache_path = cfg.CACHE_PATH
        # 读取数据时batch_size的大小
        self.batch_size = cfg.BATCH_SIZE
        # 输入图像的大小
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        # 类别与index对应起来
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        # 进行数据增强时的对称操作
        self.flipped = cfg.FLIPPED
        # 读取数据时的index
        self.cursor = 0
        # 读取数据时记录epoch的值，在打印时打印出来
        self.epoch = 1
        # 存放ground truth label
        self.gt_labels = None
        # 准备数据，制作成pkl文件
        self.prepare()

    def get(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        # 加载label
        gt_labels = self.load_labels()
        # 左右翻转图像，这里只对label进行翻转，真正图像的翻转在读取数据时进行
        if self.flipped:
            print('Appending flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - \
                                                                  gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        # 进行打乱
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        # pkl文件路径
        cache_file = os.path.join(self.cache_path, 'gt_labels.pkl')

        # 判断pkl文件是否存在，若存在直接读取pkl文件中的内容
        if os.path.isfile(cache_file):
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        # 如果不存在pkl文件则制作pkl文件， 建立一个pkl文件
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # 得到数据的名称
        txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        with open(txtname, 'r') as f:
            image_index = [x.strip() for x in f.readlines()]

        # 存放每个数据的属性
        gt_labels = []
        for index in image_index:
            # 加载每个图像的label和每个图像中的目标数目
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            # 以字典的形式存放每一个数据：imname存放图像的名称、label的shape为(7, 7, 25)、flipped是否左右反转图像
            # 注意：在制作pkl时并不对原始图像数据进行翻转，只是用flipped进行记录是否翻转的标识
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        # 写入pkl文件
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        :param index:每个图片的名称
        :return: ground truth和每个图像的目标数目
        """
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, 25))
        # 打开xml文件，读取里面的内容（目标的坐标和名称）
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)


class Tensor(object):
    def __init__(self, is_training=True):
        # 下面这些参数在Data模块介绍过
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        # 这些scale是在构建loss函数时各个损失的比重，使损失更加均衡
        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        # base学习率
        self.learning_rate = cfg.LEARNING_RATE
        # batch size
        self.batch_size = cfg.BATCH_SIZE
        # leaky relu中的alpha参数
        self.alpha = cfg.ALPHA

        # 坐标偏移量
        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                              (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        # 网络的输入
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        # 网络的输出，(batch_size, 1470)
        self.logits = self.build_network(self.images, num_outputs=self.output_size,
                                         alpha=self.alpha, is_training=is_training)

        # label
        self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
        # 损失，这里把所有的损失集中起来了
        self.loss_layer(self.logits, self.labels)
        # 得到全部损失
        self.total_loss = tf.losses.get_total_loss()
        # 记录总损失，在tensorboard中查看
        tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self, images, num_outputs, alpha, keep_prob=0.5, is_training=True, scope='yolo'):
        """
        :param images:输入数据
        :param num_outputs: 输出的shape
        :param alpha: leaky relu的alpha参数
        :param keep_prob: dropout参数
        :param is_training: 在训练使dropout为0.5，在测试使dropout为1
        :param scope: 管理tensor
        :return: 输出(batch_size, 1470)
        """
        with tf.variable_scope(scope):
            # slim.arg_scope()的作用是在调用list中函数时，自动传入后面的那些参数，从而简化代码
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        with tf.variable_scope(scope):
            # 将坐标从中心点与宽高(x_center, y_center, w, h)转换成坐标的形式(x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # 计算左上点与右下点坐标
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # 交集部分面积
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # 两块面积总和
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        """
        :param predicts:预测的结果
        :param labels: 真实label
        :param scope: tensor管理
        :return: 没有return，但是loss都被保存到tf.losses中了
        """
        with tf.variable_scope(scope):
            # 得到预测的类别、置信度、框的坐标（这里不是真正的坐标，需要经过转换，详情看论文）
            predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                         [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                        [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:],
                                       [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # label中的置信度（有目标就是1，没有就是0）、目标的坐标、类别
            response = tf.reshape(labels[..., 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[..., 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]

            # 计算坐标偏移，使预测框与真实框在同一个量级下进行对比
            offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32),
                                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            predict_boxes_tran = tf.stack([(predict_boxes[..., 0] + offset) / self.cell_size,
                                           (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                                           tf.square(predict_boxes[..., 2]),
                                           tf.square(predict_boxes[..., 3])], axis=-1)

            # 计算预测的框与真实框的iou
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # 将那些与真实框重合度高的位置置为1，也就是有目标的置为1
            object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # 没有目标的置为0
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            # 真实坐标的计算
            boxes_tran = tf.stack([boxes[..., 0] * self.cell_size - offset,
                                   boxes[..., 1] * self.cell_size - offset_tran,
                                   tf.sqrt(boxes[..., 2]),
                                   tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                        name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                         name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                           name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                        name='coord_loss') * self.coord_scale

            # 将损失存放在tf.losses中
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            # 记录summary
            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


class YOLO(object):
    def __init__(self, args):
        # 构建数据集
        self.data = Data()
        # 预训练模型路径
        self.weights_file = cfg.WEIGHTS_FILE
        # 迭代次数
        self.max_iter = cfg.MAX_ITER
        # 初始化学习率
        self.initial_learning_rate = cfg.LEARNING_RATE
        # 每decay_steps下降一次学习率
        self.decay_steps = cfg.DECAY_STEPS
        # 学习率下降率
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        # summary文件保存路径
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER

        # 模型识别的类别
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        # 模型输入图像的大小
        self.image_size = cfg.IMAGE_SIZE
        # 模型输出特征图的大小
        self.cell_size = cfg.CELL_SIZE
        # 每个cell预测两个框
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        # 置信度的阈值
        self.threshold = cfg.THRESHOLD
        # iou的阈值
        self.iou_threshold = cfg.IOU_THRESHOLD
        # 模型的输出是（1，1470）的分为三个部分：类别、置信度和框预测
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        # 构建默认图
        graph = tf.Graph()
        with graph.as_default():
            # 构建网络模型
            self.net = Tensor(is_training=args.is_training)
            # 保存模型类
            self.saver = tf.train.Saver()
            # 模型保存地址
            self.ckpt_file = cfg.MODEL
            # 将之前的summary统一化管理
            self.summary_op = tf.summary.merge_all()
            # 构建summary文件写入器
            self.writer = tf.summary.FileWriter(cfg.SUMMARY_PATH)

            # learning中的参数，记录step
            self.global_step = tf.train.create_global_step()
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, self.decay_steps,
                                                            self.decay_rate, self.staircase, name='learning_rate')

            # 构建优化器
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.net.total_loss, self.optimizer, global_step=self.global_step)

            # 配置gpu
            gpu_options = tf.GPUOptions()
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config, graph=graph)

            # is_training判断是否是进行训练，如果为真进行训练，为假进行测试
            if args.is_training:
                # 初始化模型，如果不进行全局初始化会报错
                self.sess.run(tf.global_variables_initializer())
                print('Restoring pre_trained weight...')
                # 加载预训练模型，如果想自己训练模型的画就不需要加载预训练模型，不过这样的话会特别慢，建议不要这样
                self.saver.restore(self.sess, self.weights_file)
            else:
                # 加载自己的训练模型
                print("Restoring my trained model...")
                self.saver.restore(self.sess, self.ckpt_file)

            self.writer.add_graph(self.sess.graph)

    def train(self):
        """
        开始训练
        """
        for step in range(1, self.max_iter + 1):
            # 加载数据
            images, labels = self.data.get()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}
            summary_str, loss, _ = self.sess.run([self.summary_op, self.net.total_loss, self.train_op],
                                                 feed_dict=feed_dict)
            self.writer.add_summary(summary_str, step)

            if step % self.save_iter == 0:
                print('Epoch:{}, Step:{}, Loss:{}'.format(self.data.epoch, int(step), loss))
                self.saver.save(self.sess, self.ckpt_file)

    def detect(self, img):
        # 将读取的图像数据转换成yolo的输入形式
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        # 将图像数据传入到yolo中得到结果
        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)
        return result

    def detect_from_cvmat(self, inputs):
        # 得到yolo的输出(1, 1470)
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        # interpret_output()：对logits的输出进行处理得到标注框、类别和置信度
        results.append(self.interpret_output(net_output[0]))
        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2],
                            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(np.reshape(offset, [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,
                                         axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],
                           boxes_filtered[i][0],
                           boxes_filtered[i][1],
                           boxes_filtered[i][2],
                           boxes_filtered[i][3],
                           probs_filtered[i]])
        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA
            cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5],
                        (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, lineType)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--is_training", default=False, type=bool)
    parser.add_argument("--test_data", default='1.jpg', type=str)
    parser.add_argument("--data_type", default='image', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    # 模型构建
    yolo = YOLO(args)

    if args.is_training:
        # 模型训练
        yolo.train()
    else:
        # 文件路径
        file_path = os.path.join(cfg.TEST_DATA, args.test_data)
        if args.data_type == 'image':
            # 读取文件
            image = cv2.imread(file_path)
            # 检测目标
            result = yolo.detect(image)
            # 在原图上画出框
            yolo.draw_result(image, result)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        elif args.data_type == 'video':
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            while ret:
                result = yolo.detect(frame)
                yolo.draw_result(frame, result)
                cv2.imshow("video", frame)
                cv2.waitKey(10)
                ret, frame = cap.read()
        else:
            raise print("file type error...")


if __name__ == '__main__':
    main()
