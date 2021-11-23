import glob
import time
from utils.datasets import *
from utils.utils import plot_one_box
import os
import cv2
import requests
import shutil
from video_demo.request_api import SoftFPN
import numpy
from pathlib import Path
from video_demo.utils import convert_det_dict, plot_one_box


class Classifier(object):
    def __init__(self, root_path, save_path, detect_type, track_type):
        self.root_path = root_path
        self.save_path = save_path
        self.img_paths = self._get_all_files(root_path)
        self.classifier = SoftFPN(detect_type=detect_type)
        self.track_type = track_type

    def _get_all_files(self, root_path):
        img_paths, img_extentions = [], ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        for folder_name in os.listdir(root_path):
            current_folder = root_path / folder_name
            for sub in current_folder.glob("**/*"):
                if sub.suffix in img_extentions:
                    img_paths.append(str(sub))
        return img_paths

    def detect_img(self):
        imgs = []
        for i in range(0, len(self.img_paths), 16):
            if (len(self.img_paths) - i) < 16:
                imgs_path = self.img_paths[i:]
            else:
                imgs_path = self.img_paths[i:i+16]
            for img_path in imgs_path:
                img_original = cv2.imread(img_path)
                imgs.append(img_original)

            label_dict = self.classifier(imgs)
            labels = label_dict["class"]["label"]
            scores = label_dict["class"]["score"]

            for index, label in enumerate(labels):
                if label != self.track_type:
                    save_folder = os.path.join(self.save_path, label)
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    save_current_img_path = os.path.join(save_folder, f"{(scores[index])[:4]}_{os.path.basename(imgs_path[index])}")
                    shutil.copyfile(imgs_path[index], save_current_img_path)
                    # print(save_current_img_path)
            imgs = []


if __name__ == "__main__":
    root_path = Path("/mnt2/shy2/reid_dataset/person/public/cmdm/msmt17/query")
    save_path = Path("/mnt2/sjh/农行data/reid_badcase/msmt17")
    detect_type = "mob2staff8"
    track_type = "person"
    detector = Classifier(root_path, save_path, detect_type, track_type)
    detector.detect_img()
