import random
import shutil
import os
import glob
import random


def move_file(img_path, dst_path):
    for dir_name in os.listdir(img_path):
        current_img_path = os.path.join(img_path, dir_name)
        img_dirs = glob.glob(os.path.join(current_img_path, "*.jpg"))
        for img_dir in img_dirs:
            img_name = os.path.basename(img_dir)
            if img_name.startswith('take'):
                dst_img_path = os.path.join(dst_path, img_name)
                shutil.copy(img_dir, dst_img_path)


def move_video2folder(video_dirs):
    for video_dir in video_dirs:
        print(video_dir)
        video_name = os.path.basename(video_dir)
        camera_num = video_name.split("_")[0]
        camera_folder = os.path.join(os.path.dirname(video_dir), camera_num)
        if not os.path.exists(camera_folder):
            os.makedirs(camera_folder)
        dst_path = os.path.join(camera_folder, video_name)
        shutil.move(video_dir, dst_path)


def random_pick_video(video_path, dst_folder):
    video_folders = os.listdir(video_path)
    for video_folder in video_folders:
        cur_video_folder = os.path.join(video_path, video_folder)
        video_dirs = glob.glob(os.path.join(cur_video_folder, "*.mp4"))
        print(len(video_dirs))
        a = random.randint(0, len(video_dirs)-1)
        picked_video_dir = video_dirs[a]

        video_name = os.path.basename(picked_video_dir)
        dst_path = os.path.join(dst_folder, video_name)
        shutil.copy(picked_video_dir, dst_path)


def pick_img(img_path, dst_folder):

    img_folders = []
    for folder_name in os.listdir(img_path):
        if folder_name[0] == "C":
            img_folders.append(folder_name)
    for img_folder in img_folders:
        img_dir = os.path.join(img_path, img_folder)
        img_classes = os.listdir(img_dir)
        for img_class in img_classes:
            src_img_folder = os.path.join(img_dir, img_class)
            img_dirs = glob.glob(os.path.join(src_img_folder, "*.jpg"))
            dst_img_folder = os.path.join(dst_folder, img_class)
            for img_dir in img_dirs:
                img_name = os.path.basename(img_dir)
                dst_img = os.path.join(dst_img_folder, img_name)
                shutil.copy(img_dir, dst_img)


def check_file(img_path, dst_path):
    import cv2
    img_folders = os.listdir(img_path)
    for img_folder in img_folders:
        current_img_folder = os.path.join(img_path, img_folder)
        img_dirs = glob.glob(os.path.join(current_img_folder, "*.jpg"))
        for img_dir in img_dirs:
            img_name = os.path.basename(img_dir)
            img_folder_dst = os.path.join(dst_path, img_folder)
            if not os.path.exists(img_folder_dst):
                os.makedirs(img_folder_dst)
                xml_folder_dst = img_folder_dst.replace("JPEGImages", "Annotations")
                os.makedirs(xml_folder_dst)
            img_dir_dst = os.path.join(img_folder_dst, img_name)
            xml_dir_dst = img_dir_dst.replace("JPEGImages", "Annotations").replace("jpg", "xml")
            print(img_dir)
            # img = cv2.imread(img_dir)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # xml_dir = img_dir.replace("JPEGImages", "Annotations").replace("jpg", "xml")
            xml_dir = "/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/轮椅拐杖数据集/拐杖数据集_已修正/Annotations/data/{}".format(img_name.replace("jpg", "xml"))
            shutil.copyfile(img_dir, img_dir_dst)
            shutil.copyfile(xml_dir, xml_dir_dst)


def move_file2all(img_path, dst_path):
    folder_lists = os.listdir(img_path)
    for folder_list in folder_lists:
        current_img_folder = os.path.join(img_path, folder_list)
        img_dirs = glob.glob(os.path.join(current_img_folder, "*.jpg"))
        for img_dir in img_dirs:
            img_name = os.path.basename(img_dir)


if __name__ == "__main__":
    img_path = "/mnt2/shy2/x_ray/x光机图像_64393"
    dst_path = "/mnt2/shy2/x_ray/x光机原始图像_无目录"
    move_file(img_path, dst_path)


