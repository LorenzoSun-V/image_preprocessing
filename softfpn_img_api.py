import glob
import time
from utils.datasets import *
from utils.utils import plot_one_box
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os
import cv2
import requests
import shutil
# import sys; sys.path.append("..")
import numpy
from pathlib import Path
from video_demo.utils import convert_det_dict, plot_one_box


class SoftFPN():
    def __init__(self, detect_type, demo=False):
        self.detect_type = detect_type
        # self.HOST = 'http://192.167.10.5:8888'
        self.HOST = 'http://192.167.10.10'
        self.login_url = "%s/login" % self.HOST
        self.upload_url = "%s/upload/image" % self.HOST
        self.detect_url = "%s/detect/api/img" % self.HOST
        self.demo = demo

    def get_sess(self):
        userAgent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"
        header = {'User-Agent': userAgent, }
        data = {'username': 'admin', 'password': 'admin123456'}

        # 通过session模拟登录，每次请求带着session
        self.sess = requests.Session()

        f = self.sess.post(self.login_url, data=data, headers=header)
        # print(json.loads(f.text))

    def __call__(self, detect_data, frame_id=None):
        self.get_sess()
        if isinstance(detect_data, numpy.ndarray):
            frame = detect_data
            # 图片的字节流
            img_data = cv2.imencode(".jpg", frame)[1].tobytes()
            img_name = str(frame_id) + ".jpg"
        elif isinstance(detect_data, str):
            if os.path.exists(detect_data):
                img_path = detect_data
                img_name = img_path.rsplit("/", 1)[-1]
                img_data = open(img_path, 'rb')
                frame = cv2.imread(img_path)
            else:
                print("The image file does not found!")
                return
        else:
            return
        files = {'image': (img_name, img_data, 'image/png')}
        data = {'detect_type': self.detect_type, }
        # 上传图片
        resp = self.sess.post(self.upload_url, data=data, files=files)
        resp = resp.json()

        if resp["task_id"]:
            task_id = resp["task_id"]
            detect_data = self.query_result(task_id)

        return frame, detect_data

    def query_result(self, task_id):
        detect_url = "%s/%s" % (self.detect_url, task_id)
        data = {"detect_type": self.detect_type}
        resp = self.sess.post(detect_url, data=data)
        # print("resp")
        # print(resp.text)
        resp = resp.json()["detect_data"]
        return resp


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

    def detect_img(self):
        for img_path in self.img_paths:
            img_original = cv2.imread(img_path)
            _, det_data = self.detector(img_original)  # _ is frame which we don't need
            if len(det_data) > 0:
                # 存原始小图
                save_current_img_path = str(self.save_path / "original" / os.path.basename(img_path))
                shutil.copyfile(img_path, save_current_img_path)

                det_dict = convert_det_dict(det_data)
                print(det_dict)
                tlbrs = det_dict[self.track_type]["tlbrs"]
                for tlbr in tlbrs:
                    plot_one_box(img_original, tlbr, self.track_type, color=(0, 0, 255))
                # 存画框小图
                save_current_img_path = str(self.save_path / "result" / os.path.basename(img_path))
                cv2.imwrite(save_current_img_path, img_original)
            # else:
            #     save_current_img_path = str(self.save_path / "right" / os.path.basename(img_path))
            #     cv2.imwrite(save_current_img_path, img_original)


if __name__ == "__main__":
    root_path = Path("/mnt/shy/农行POC/abc_data/第六批1018/wheelchair/JPEGImages")
    save_path = Path("/mnt/shy/农行POC/abc_data/第六批1018/wheelchair/result")
    detect_type = "wheelchair_crutch"
    track_type = "wheelchair"  # track_type => the classification we focus on
    detector = Detector(root_path, save_path, detect_type, track_type)
    detector.detect_img()
