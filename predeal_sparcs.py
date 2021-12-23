import shutil

import gdal
import os
import glob
import numpy as np
from PIL import Image


#  读取tif数据集
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def to_mydataset(root_path):
    img_dirs = glob.glob(os.path.join(root_path, "*_photo.png"))
    save_path = "/home/user/Downloads/labels/"
    # print(len(img_dirs))
    for img_dir in img_dirs:
        img_name = img_dir[:-10]
        # print(img_name)
        label_path = img_name + "_labels.tif"
        label_name = os.path.basename(label_path).replace('tif', 'png')
        save = save_path + label_name
        width, height, bands, data, geotrans, proj = readTif(label_path)
        img = Image.fromarray(data).save(save)


def train_test_split(root_path):
    save_image_path = "/mnt2/sjh/seg_data/mySPARCS/images"
    save_label_path = "/mnt2/sjh/seg_data/mySPARCS/masks"
    save_vis_path = "/mnt2/sjh/seg_data/mySPARCS/vis"
    image_path = os.path.join(root_path, "images")
    label_path = os.path.join(root_path, "new_labels")
    vis_path = os.path.join(root_path, "vis")
    img_dirs = glob.glob(os.path.join(image_path, "*.png"))
    import random
    random.seed(666)
    random.shuffle(img_dirs)
    test_img_dirs = img_dirs[:8]
    train_img_dirs = img_dirs[8:]
    for i, img_dir in enumerate(train_img_dirs):
        img_name = os.path.basename(img_dir)
        label_name = img_name.replace('photo', 'labels')
        label_dir = os.path.join(label_path, label_name)
        vis_dir = os.path.join(vis_path, img_name)
        save_image_dir = os.path.join(save_image_path, f"train/{i}.png")
        save_label_dir = os.path.join(save_label_path, f"train/{i}.png")
        save_vis_dir = os.path.join(save_vis_path, f"train/{i}.png")
        shutil.copyfile(img_dir, save_image_dir)
        shutil.copyfile(label_dir, save_label_dir)
        shutil.copyfile(vis_dir, save_vis_dir)

    for i, img_dir in enumerate(test_img_dirs):
        img_name = os.path.basename(img_dir)
        label_name = img_name.replace('photo', 'labels')
        label_dir = os.path.join(label_path, label_name)
        vis_dir = os.path.join(vis_path, img_name)
        save_image_dir = os.path.join(save_image_path, f"test/{i}.png")
        save_label_dir = os.path.join(save_label_path, f"test/{i}.png")
        save_vis_dir = os.path.join(save_vis_path, f"test/{i}.png")
        shutil.copyfile(img_dir, save_image_dir)
        shutil.copyfile(label_dir, save_label_dir)
        shutil.copyfile(vis_dir, save_vis_dir)


if __name__ == "__main__":
    # to_mydataset("/home/user/Downloads/sparcs_data_L8")
    train_test_split("/home/user/Downloads")
