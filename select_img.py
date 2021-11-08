import os
import glob
import shutil


img_path = "/home/user/PycharmProjects/classification/mobilenet/dataset/train_all_7/3-person"
img_dirs = glob.glob(os.path.join(img_path, "*.jpg"))
search_path = "/mnt/shy/sjh/mobilenet/dataset/train_all_7/3-person"
dst_path = "/home/user/PycharmProjects/classification/mobilenet/dataset/person_search"
for img_dir in img_dirs:
    img_name = os.path.basename(img_dir)
    img_search_dir = os.path.join(search_path, img_name)
    if not os.path.exists(img_search_dir):
        print(img_search_dir)
        img_dst_path = os.path.join(dst_path, img_name)
        shutil.copy(img_dir, img_dst_path)
