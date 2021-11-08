import os
import glob
import shutil
import cv2


def delete_file(src_path, dst_path):
    # 如果文件在src_path中存在，则在dst_path中删除该文件
    img_dirs = glob.glob(os.path.join(src_path, "*.jpg"))
    for img_dir in img_dirs:
        img_name = os.path.basename(img_dir)
        dst_img_path = os.path.join(dst_path, img_name)
        if os.path.exists(dst_img_path):
            print(dst_img_path)
            os.remove(dst_img_path)


def delete_file2(img_path):
    img_dirs = glob.glob(os.path.join(img_path, "*.jpg"))
    for i, img_dir in enumerate(img_dirs):
        img = cv2.imread(img_dir)
        print("{} / {}".format(i, len(img_dirs)-1))
        h, w, _ = img.shape
        if h < 40 or w < 40:
            os.remove(img_dir)


if __name__ == "__main__":
    # src_path = "/mnt/shy/农行POC/第三批0716/part1_cut_video/C92/security_staff"
    # dst_path = "/home/user/data/农行/测试数据/第三批/security_staff"
    # delete_file(src_path, dst_path)

    img_path = "/mnt2/private_data/郑州云踪e/train_data/small_img/validation"
    img_folders = ["0-bank_staff_female", "2-bank_staff_male"]
    for img_folder in img_folders:
        current_img_path = os.path.join(img_path, img_folder)
        delete_file2(current_img_path)
