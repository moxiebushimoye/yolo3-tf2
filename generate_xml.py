import os
import shutil

from PIL import Image

from utils.create_xml import VOC_Sample_Generator
from yolo import YOLO
import cv2
import numpy as np
from utils.utils import (cvtColor, preprocess_input,resize_image)
import tensorflow as tf

###初始化yolo模型 - 全局变量###
yolo=YOLO()
#############################


def get_xml(img_path, image, xmlsave_path,move_path):
    '''
    img_path:图像文件路径
    image:PIL读取的图片对象
    xmlsave_path：生成的xml文件存储的文件夹名
    运行生成图像标注文件
    '''
    img_name = img_path.split("/")[-1]
    xml_name = img_name.split('.')[0]
    try:
        img = cv2.imread(img_path)
        img_width = img.shape[0]
        img_height = img.shape[1]
        img_depth = img.shape[-1]
    except Exception:
        print(img_path)
    image = cvtColor(image)

    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize        #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data = resize_image(image, (yolo.input_shape[1], yolo.input_shape[0]), yolo.letterbox_image)
    # ---------------------------------------------------------#
    #   添加上batch_size维度，并进行归一化
    # ---------------------------------------------------------#
    image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
    ### keras版本 ###
    # out_boxes, out_scores, out_classes = self.sess.run(
    #     [self.boxes, self.scores, self.classes],
    #     feed_dict={
    #         self.yolo_model.input: image_data,
    #         self.input_image_shape: [image.size[1], image.size[0]],
    #         K.learning_phase(): 0
    #     })
    #
    # ---------------------------------------------------------#
    #   将图像输入网络当中进行预测！
    # ---------------------------------------------------------#
    input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
    out_boxes, out_scores, out_classes = yolo.get_pred(image_data, input_image_shape)

    # print(out_classes.numpy().size)
    voc = VOC_Sample_Generator()
    if out_classes.numpy().size != 0:
        voc.add_filename(img_name)
        voc.add_size(img_width, img_height, img_depth)
        for i, c in list(enumerate(out_classes)):
            predicted_class = yolo.class_names[int(c)]
            # box = out_boxes[i]
            top, left, bottom, right = out_boxes[i]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            voc.build_object(predicted_class, left, top, right, bottom)
        voc.build(xmlsave_path + xml_name + ".xml")
    else:
        shutil.move(img_path,move_path)




def xml_main(target_dir,xml_path,move_dir):
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
    # img_path = "appleimg/apple1.jpg"
    if not os.path.exists(move_dir):
        os.makedirs(move_dir)
    imgpath_list=os.listdir(target_dir)
    for img_name in imgpath_list:
        image_path = os.path.join(target_dir, img_name)
        image = Image.open(image_path)
        move_path=os.path.join(move_dir,img_name)
        get_xml(image_path, image,xml_path, move_path)


if __name__ == '__main__':
    # 存储待标注图像的文件夹路径
    target_dir = "auto_label/images/"
    # 生成标志文件的存储路径
    xml_path = "auto_label/xml_anno/"
    move_dir="auto_label/noxml_img/"
    xml_main(target_dir,xml_path,move_dir)
