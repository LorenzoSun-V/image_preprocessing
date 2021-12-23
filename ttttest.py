import os
import glob
import gdal
import numpy as np
from PIL import Image


a = "/mnt2/sjh/seg_data/mySPARCS/masks_UNet/7.png"
b = np.array(Image.open(a))
# b[b != 2]
img = np.unique(np.array(Image.open(a)))

label_path = "/home/user/Downloads/labels"
save_label_path = "/home/user/Downloads/new_labels/"
label_dirs = glob.glob(os.path.join(label_path, '*.png'))
for label_dir in label_dirs:
    label_name = os.path.basename(label_dir)
    label = np.array(Image.open(label_dir))
    label[label == 0] = 20
    label[label == 1] = 20
    label[label == 2] = 100
    label[label == 3] = 100
    label[label == 4] = 100
    label[label == 5] = 200
    label[label == 6] = 100
    label[label == 100] = 0
    label[label == 200] = 1
    label[label == 20] = 2
    current_save_path = save_label_path + label_name
    Image.fromarray(label).save(current_save_path)
    print(label_dir)


# pixel_dict = {20480: 4, 20512: 4, 23552: 3, 28672: 5, 36864: 4, 36896: 4, 53248: 5, 53280: 5}
#
# path = "/home/user/Downloads/sparcs_data_L8"
# img_dirs = glob.glob(os.path.join(path, '*_labels.tif'))
# print(len(img_dirs))
#
# img_dir = "/home/user/Downloads/sparcs_data_L8/LC80010812013365LGN00_18_qmask.tif"
# img_dir2 = "/home/user/Downloads/sparcs_data_L8/LC80020622013244LGN00_32_qmask.tif"
# img = gdal.Open(img_dir)
# img2 = gdal.Open(img_dir2)
# width = img.RasterXSize
# height = img.RasterYSize
# bands = img.RasterCount
# img = img.ReadAsArray(0, 0, width, height)
# img2 = img2.ReadAsArray(0, 0, width, height)
# for pixel in pixel_dict:
#     img[img == pixel] = pixel_dict[pixel]
#     img2[img2 == pixel] = pixel_dict[pixel]
#
# img_save = "/home/user/Downloads/sparcs_data_L8/LC80010812013365LGN00_18_labels.png"
# img_save2 = "/home/user/Downloads/sparcs_data_L8/LC80020622013244LGN00_32_labels.png"
#
# img = Image.fromarray(img).save(img_save)
# img2 = Image.fromarray(img2). save(img_save2)
#
# print(img)
# print(img2)

