import random
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import cv2
import math


class Augmentation:
    def __init__(self):
        pass

    def rotate(self, image, angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-180, 180])  # -180~180随机选一个角度旋转
        if isinstance(angle, list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        image = tf.to_tensor(image)
        return image

    def flip(self, image):  # 水平翻转和垂直翻转
        if random.random() > 0.5:
            image = tf.hflip(image)
        if random.random() < 0.5:
            image = tf.vflip(image)
        image = tf.to_tensor(image)
        return image

    def randomResizeCrop(self, image, scale=(0.6, 1.0),
                         ratio=(1, 1)):  # scale表示随机crop出来的图片会在的0.3倍至1倍之间，ratio表示长宽比
        h_image = image.size[0]
        img = np.array(image)
        # h_image, w_image = img.shape
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        image = tf.to_tensor(image)
        return image

    def adjustContrast(self, image):
        factor = transforms.RandomRotation.get_params([0, 10])  # 这里调增广后的数据的对比度
        image = tf.adjust_contrast(image, factor)
        # mask = tf.adjust_contrast(mask,factor)
        image = tf.to_tensor(image)
        return image

    def adjustBrightness(self, image):
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_brightness(image, factor)
        # mask = tf.adjust_contrast(mask, factor)
        image = tf.to_tensor(image)
        return image

    def centerCrop(self, image, size=None):  # 中心裁剪
        if size == None: size = image.size  # 若不设定size，则是原图。
        image = tf.center_crop(image, size)
        image = tf.to_tensor(image)
        return image

    def adjustSaturation(self, image):  # 调整饱和度
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_saturation(image, factor)
        # mask = tf.adjust_saturation(mask, factor)
        image = tf.to_tensor(image)
        return image

    def random_affine(self, image, label, perspective=0.0, degrees=0.373, scale=0.898, shear=0.602, translate=0.245):
        # 随机仿射(随机偏移，随机旋转，随机放缩等整合)
        height, width = image.shape[:2]

        # Center refer yolov5's mosaic aug
        C = np.eye(3)
        C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees) / math.pi * 180  # 增加将弧度 转成角度
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation float，先中心偏移，再进行各种操作，然后将中心转移至原始位置左右，都是随机
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (M != np.eye(3)).any():  # image changed
            image = cv2.warpAffine(image, M[:2], dsize=self.input_hw[::-1], borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.image_fill)
            label = cv2.warpAffine(label, M[:2], dsize=self.input_hw[::-1], borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.label_fill)
        else:
            # 若未变换，则直接resize，这种概率很小
            image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label


def augmentationData(image_path, option=[2, 4, 6, 7], save_dir=None):
    '''
    :param image_path: 图片的路径
    :param option: 需要哪种增广方式：
    ：1为旋转，2为翻转，3为随机裁剪并恢复原本大小，4为调整对比度，5为中心裁剪(不恢复原本大小)，6为调整亮度,7为饱和度
    :param save_dir: 增广后的数据存放的路径
    '''
    folder_name = os.path.basename(image_path)
    aug_image_savedDir = os.path.join(save_dir, folder_name)
    if not os.path.exists(aug_image_savedDir):
        os.makedirs(aug_image_savedDir)
        print('create aug img dir.....')
    aug = Augmentation()
    res = os.walk(image_path)
    images = []
    masks = []
    for root, dirs, files in res:
        for f in files:
            images.append(os.path.join(root, f))

    datas = list(images)
    num = len(datas)
    length = len(datas)

    for i, image_path in enumerate(datas):
        image = Image.open(image_path)
        print("{} / {}".format(str(i), str(length)))
        if 1 in option:
            num += 1
            image_tensor = aug.rotate(image)
            image_rotate = transforms.ToPILImage()(image_tensor).save(
                os.path.join(aug_image_savedDir, 'g_' + str(num) + '_rotate.jpg'))
        if 2 in option:
            num += 1
            image_tensor = aug.flip(image)
            image_filp = transforms.ToPILImage()(image_tensor).save(
                os.path.join(aug_image_savedDir, 'g_' + str(num) + '_filp.jpg'))
        if 3 in option:
            num += 1
            image_tensor = aug.randomResizeCrop(image)
            image_ResizeCrop = transforms.ToPILImage()(image_tensor).save(
                os.path.join(aug_image_savedDir, 'g_' + str(num) + '_ResizeCrop.jpg'))
        if 4 in option:
            num += 1
            image_tensor = aug.adjustContrast(image)
            image_Contrast = transforms.ToPILImage()(image_tensor).save(
                os.path.join(aug_image_savedDir, 'g_' + str(num) + '_Contrast.jpg'))
        if 5 in option:
            num += 1
            image_tensor = aug.centerCrop(image)
            image_centerCrop = transforms.ToPILImage()(image_tensor).save(
                os.path.join(aug_image_savedDir, 'g_' + str(num) + '_centerCrop.jpg'))
        if 6 in option:
            num += 1
            image_tensor = aug.adjustBrightness(image)
            image_Brightness = transforms.ToPILImage()(image_tensor).save(
                os.path.join(aug_image_savedDir, 'g_' + str(num) + '_Brightness.jpg'))
        if 7 in option:
            num += 1
            image_tensor = aug.adjustSaturation(image)
            image_Saturation = transforms.ToPILImage()(image_tensor).save(
                os.path.join(aug_image_savedDir, 'g_' + str(num) + '_Saturation.jpg'))


augmentationData("/home/user/PycharmProjects/classification/mobilenet/dataset/train_add_0719/4-security_staff",
                 save_dir="/home/user/PycharmProjects/classification/mobilenet/dataset/train_aug_add_0719")
