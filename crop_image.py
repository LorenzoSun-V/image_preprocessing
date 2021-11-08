import cv2
from pathlib import Path
from check_redundant_files import find_files, find_redundant_files, load_voc_xml
import os
import glob
import time
import numpy as np
# from FaceQualityAssessment import sharpness


if __name__ == '__main__':
    # widerface
    root_face = "/mnt/shy/上海智慧社区探索项目/倒地标定/1027/倒地人体框标签修正/0916/"
    save_path = "/mnt/shy/上海智慧社区探索项目/人体小图"

    label_list = ["person", "fall_person", "crouch_person"]
    # label_list = ["bank_staff", "cleaner",  "money_staff", "person", "security_staff"]
    # get all folders name
    folder_lists = os.listdir(root_face)

    for folder_name in folder_lists:
        img_dir = Path(root_face) / folder_name / "JPEGImages"
        xml_dir = Path(root_face) / folder_name / "Annotations"
        # 获取 针对二级文件夹的图片路径
        try:
            xml_files = find_files(xml_dir, "xml")
        except: 
            print("not such annotation folder : {}".format(xml_dir))
            continue
        img_files = find_files(img_dir, "jpg")
        # 检查 对应的图片和标注文件  是否存在缺失情况
        img_paths, xml_paths = find_redundant_files(xml_files, img_files)

        for img_path, xml_path in zip(img_paths, xml_paths):
            img = cv2.imread(img_path)
            img_paths = img_path.split(os.path.sep)
            # h, w = img.shape[:2]
            tree = load_voc_xml(xml_path)
            rt = tree.getroot()
            num = 1

            for obj in rt.findall("object"):
                name = obj.find("name").text
                if name in label_list:
                    bbox = obj.find("bndbox")
                    xmin = int(float(bbox.find("xmin").text))
                    ymin = int(float(bbox.find("ymin").text))
                    xmax = int(float(bbox.find("xmax").text))
                    ymax = int(float(bbox.find("ymax").text))
                    # blur = int(float(bbox.find("blur").text))
                    # has_lm = int(float(obj.find("has_lm").text))
                    width = xmax - xmin
                    height = ymax - ymin
                    # if width>35 and (width/height<4 or height/width<4):
                    left, right, top, down = xmin, xmax, ymin, ymax
                    img_crop = img[top:down, left:right]
                    img_save_path = os.path.join(save_path, name)
                    # img_save_path = os.path.join(img_dir, name)
                    if not os.path.exists(img_save_path):
                        os.makedirs(img_save_path)
                    current_name = img_paths[-1]
                    current_name = current_name.split(".")[0]
                    one_img_save_path = img_save_path + '/' + current_name + '_' + str(num) + '_' + str(int(time.time()))[-5:] + '.jpg'
                    # print(one_img_save_path)
                    try:
                        cv2.imwrite(one_img_save_path, img_crop)
                    except:
                        print(current_name)
                    num += 1