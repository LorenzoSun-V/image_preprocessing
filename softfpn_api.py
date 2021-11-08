import glob
import time
from utils.datasets import *
from utils.utils import plot_one_box
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os
import cv2
import requests
# import sys; sys.path.append("..")
import numpy


label_list2 = ['bank_staff', 'cleaner', 'money_staff', 'person', 'security_staff']
label_list = ["person", "head_shoulder", "face"]
label_list3 = ["knife", "scissors", "sharpTools", "expandableBaton", "smallGlassBottle", "electricBaton",
				  "plasticBeverageBottle", "plasticBottleWithaNozzle", "electronicEquipment", "battery",
				  "seal", "umbrella"]
label_list4 = ["Gun", "Knife", "Scissors", "Pliers", "Wrench"]
color_list = [ (0,255,0), (255,0,0),(0,0,255) ]
color_list2 = [(169,169,169),(0,255,255),(0,128,128),(130,0,75),(203,192,255)]
color_list3 = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
				  (240, 32, 160), (79, 79, 47), (147, 20, 255), (179, 222, 245), (86, 114, 255), (197, 181, 255)]
color_list4 = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]
# 存储图片
img_save = []
# 存储图片名
name_list = []


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


def detect(img_dir, save_root, detect_type):
    softfpn_xray = SoftFPN(detect_type=detect_type)
    img_name = os.path.basename(img_dir)
    frame, detect_data= softfpn_xray(img_dir)
    save_path = os.path.join(save_root, img_name)
    if len(detect_data):
        for i in range(len(detect_data)):
            object_boxs = detect_data[i][1]
            attribute = detect_data[i][0]
            for j in range(len(object_boxs)):
                xmin = object_boxs[j][0]
                ymin = object_boxs[j][1]
                xmax = object_boxs[j][2]
                ymax = object_boxs[j][3]
                # color = color_list[label_list.index(name)]
                object_box = (xmin, ymin, xmax, ymax)
                color = color_list3[label_list3.index(attribute)]
                plot_one_box(object_box, frame, label=attribute, color=color, line_thickness=2)
    else:
        pass
    cv2.imwrite(save_path, frame)
        # cv2.imshow("fff", frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    t = time.time()
    path = "/mnt1/shy1/test_shequ/fall_demo/badcase_merge_0913"
    save_root = "/mnt1/shy1/test_shequ/fall_demo/badcase_merge_0913/demo_0922"
    # img_folders = os.listdir(path)
    # for img_folder in img_folders:
    #     current_img_folder = os.path.join(path, img_folder)
    #     img_dirs = glob.glob(os.path.join(current_img_folder, "*.*"))
    #     for img_dir in img_dirs:
    #         print(img_dir)
    #         detect_type = 'xray12'
    #         detect(img_dir, save_root, detect_type)

    # hand_person_hs_face
    img_dirs = glob.glob(os.path.join(path, "*.mp4"))
    for img_dir in img_dirs:
        detect_type = 'fall_person_hs'
        detect(img_dir, save_root, detect_type)

        print(time.time() - t)
