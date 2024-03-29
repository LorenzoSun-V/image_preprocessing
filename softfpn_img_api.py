# import glob
# import time
from utils.datasets import *
from utils.utils import plot_one_box
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os
import cv2
# import requests
import shutil
from video_demo.request_api import SoftFPN
# import numpy
from pathlib import Path
from video_demo.utils import convert_det_dict, plot_one_box


class Detector(object):
    def __init__(self, root_path, save_path, detect_type, track_type):
        self.root_path = root_path
        self.save_path = save_path
        self.img_paths, self.video_paths = self._get_all_files(root_path)
        self.detector = SoftFPN(detect_type=detect_type)
        self.track_type = track_type

    def _get_all_files(self, root_path):
        img_paths, img_extentions = [], ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        video_paths, video_extentions = [], ['.mp4', '.MP4', '.mpeg', ".MPEG"]
        for sub in root_path.glob("**/*"):
            if sub.suffix in img_extentions:
                img_paths.append(str(sub))
            elif sub.suffix in video_extentions:
                video_paths.append(str(sub))
            else:
                raise RuntimeError(f"{sub} not found")
        return img_paths, video_paths

    def detect_group(self, img_count, imgs, img_num):
        det_datas = self.detector(imgs)

        dec_dict_list = []
        for det_data in det_datas:
            det_dict = convert_det_dict(det_data)
            dec_dict_list.append(det_dict)

        for index, det_dict in enumerate(dec_dict_list):
            if self.track_type in det_dict.keys():
                # print(f"{img_count - (img_num - index)} has detected")
                # 存原始小图
                save_current_img_path = str(
                    self.save_path / "detect_original" / os.path.basename(self.img_paths[img_count - (img_num - index)]))
                if not os.path.exists(os.path.dirname(save_current_img_path)):
                    os.mkdir(os.path.dirname(save_current_img_path))
                shutil.copyfile(self.img_paths[img_count - (img_num - index)], save_current_img_path)

                tlbrs = det_dict[self.track_type]["tlbrs"]
                for tlbr in tlbrs:
                    plot_one_box(imgs[index], tlbr, self.track_type, color=(0, 0, 255))
                # 存画框小图
                save_current_img_path = str(
                    self.save_path / "detect" / os.path.basename(self.img_paths[img_count - (img_num - index)]))
                if not os.path.exists(os.path.dirname(save_current_img_path)):
                    os.mkdir(os.path.dirname(save_current_img_path))
                cv2.imwrite(save_current_img_path, imgs[index])
            else:
                # print(f"{img_count - (img_num - index)} has no detected")
                save_current_img_path = str(
                    self.save_path / "no_detect" / os.path.basename(self.img_paths[img_count - (img_num - index)]))
                if not os.path.exists(os.path.dirname(save_current_img_path)):
                    os.mkdir(os.path.dirname(save_current_img_path))
                cv2.imwrite(save_current_img_path, imgs[index])

    def detect_img(self):
        imgs = []
        img_count = 0
        final_cur = img_count
        for i, img_path in enumerate(self.img_paths):
            print(f"{i+1} / {len(self.img_paths)}")
            img = cv2.imread(img_path)
            imgs.append(img)
            img_count += 1

            if len(self.img_paths) - final_cur < 16:
                if img_count == len(self.img_paths):
                    self.detect_group(img_count, imgs, len(self.img_paths) - final_cur)
                    break
            else:
                if (img_count % 16 == 0):
                    self.detect_group(img_count, imgs, 16)
                    final_cur = img_count
                    imgs = []

        # self.detect_group(img_count, imgs, len(self.img_paths) - img_count)
        # imgs = []


if __name__ == "__main__":
    root_path = Path("/mnt2/private_data/上海银行/第一批1118/轮椅拐杖图片")
    save_path = Path("/mnt2/private_data/上海银行/第一批1118/test_folder")
    detect_type = "wheelchair_crutch" # motorcycle
    track_type = "crutch"  # track_type => the classification we focus on
    detector = Detector(root_path, save_path, detect_type, track_type)
    detector.detect_img()