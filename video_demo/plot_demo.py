'''
Author: shy
Description:
LastEditTime: 2021-09-28 14:06:07
'''
import os
import cv2
import time
import glob
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import logzero
from logzero import logger as log
from pprint import pprint
from ipdb import set_trace as pause
from pathlib import Path, PosixPath

from request_api import SoftFPN
from video import FileVideoStream
from profiler import Profiler
from utils import checkfolder, convert_det_dict, crop_im, plot_one_box, if_inPoly
from shapely import geometry
from PIL import Image,ImageDraw, ImageFont


def draw_zh_cn(frame, string, color, position, font_size=20):
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    draw = ImageDraw.Draw(pil_im)
    size_font = ImageFont.truetype(
            "/mnt/shy/sjh/NotoSansCJK-Black.ttc", font_size)
    draw.text(position, string, color, font=size_font)
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return img


label_list = ["person", "head_shoulder", "face"]
label_list2 = ['bank_staff', 'cleaner', 'money_staff', 'person', 'security_staff']
label_list3 = ['bank_staff_vest', 'cleaner', 'money_staff', 'person', 'security_staff', 'bank_staff_shirt',
               'bank_staff_coat']
color_list = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
color_list2 = [(169, 169, 169), (0, 255, 255), (0, 128, 128), (130, 0, 75), (203, 192, 255)]
color_list3 = [(169, 169, 169), (0, 255, 255), (0, 128, 128), (130, 0, 75), (203, 192, 255), (0, 165, 255),
               (235, 206, 135)]
color_dict3 = {k: v for k, v in zip(label_list3, color_list3)}


def creat_log_dir(cfg):
    prefix = time.strftime('%Y-%m%d-%H%M-%S', time.localtime(time.time()))
    log_dir = Path(cfg.log_dir) / prefix
    checkfolder(log_dir)
    log_path = str(log_dir / "track_log.txt")
    logzero.logfile(log_path, maxBytes=1e6, backupCount=3)
    cfg.log_dir = log_dir

    log.info('\n=================New Log=================\n')
    log.info(cfg)
    pprint(cfg)
    log.info('\n=================New Log=================\n')


class MOT:
    def __init__(self, cfg, debug=False):
        ## 启动模型服务
        self.cfg = cfg
        # self.det_model = SoftFPN('person_hs_face')
        # softfpn_det   = SoftFPN(detect_type='bank_person_hs')
        self.det_model = SoftFPN(detect_type='bank_person_hs')
        self.cls_model = SoftFPN(detect_type='mob2staff7')

        if cfg.track_type == "person":
            self.feat_model = SoftFPN('person_feature')
        elif cfg.track_type == "head_shoulder":
            self.feat_model = SoftFPN('hs_feature')
        else:
            raise TypeError(f" wrong track type : {cfg.track_type}, please check")

    # self.tracker  = MultiTracker(cfg)

    def step(self, frame, frame_id, frame_count, video_name):
        tlbrs_roi = np.array([[[343, 25], [1042, 36], [1216, 420], [1033, 542], [366, 576], [206, 459]]], dtype=np.int32)
        cv2.polylines(frame, tlbrs_roi, True, thickness=3, color=(0, 255, 255))  # 绘制多边形
        tlbrs_roi = [(343, 25), (1042, 36), (1216, 420), (1033, 542), (366, 576), (206, 459)]
        with Profiler('detect'):
            det_data = self.det_model(frame)
        det_dict = convert_det_dict(det_data)

        have_bs = False
        have_person = False

        if self.cfg.track_type in det_dict.keys():

            # 裁剪目标小图，提取深度特征
            tlbrs = det_dict[cfg.track_type]["tlbrs"]
            ims = [crop_im(frame, tlbr) for tlbr in tlbrs]

            with Profiler('mob2staff'):
                # cls_result = [self.cls_model(im)['class']    for im in ims]
                # labels = [ i['label'] for i in cls_result]
                # scores = [ i['score'] for i in cls_result]

                label_dict = self.cls_model(ims)
                labels = label_dict["class"]["label"]
                scores = label_dict["class"]["score"]

            # 存小图
            if self.cfg.save_img:
                for i, label in enumerate(labels):
                    img_name = "{}_({}_{}).jpg".format(video_name.split(".")[0], frame_count, i)
                    save_img_path = cfg.log_dir / "img" / f"{label_list3.index(label)}-{label}" / img_name
                    dir_path = save_img_path.parent
                    if not dir_path.exists():
                        dir_path.mkdir(parents=True)
                    cv2.imwrite(str(save_img_path), ims[i])

            # 绘制检测框信息
            for i, (tlbr, label, score) in enumerate(zip(tlbrs, labels, scores)):
                if frame_id <= 175:
                    label = "bank_staff_shirt"
                text = label + " " + score[:4]
                # text = label+ " " + str(round(score,2))
                plot_one_box(frame, tlbr, text, color=color_dict3[label])
                xmin, ymin, xmax, ymax = tlbr
                center_person = int(xmin + (xmax - xmin) / 2), int(ymin + (ymax - ymin) / 2)
                print(if_inPoly(tlbrs_roi, center_person))
                if if_inPoly(tlbrs_roi, center_person):
                    if label.startswith("bank"):
                        have_bs = True
                    elif label == "person":
                        have_person = True

        if have_bs and have_person:
            plot_text = "开放中"
        elif (have_bs == True) and (have_person == False):
            plot_text = "空闲中"
        elif have_bs == False:
            plot_text = "未开放"
        print(plot_text)
        frame = draw_zh_cn(frame, f"当前状态:{plot_text}", color=(255, 255, 0), position=(343, 25),
                           font_size=30)
        if frame_id <= 120:
            if plot_text == "未开放":
                frame = draw_zh_cn(frame, f"2021-10-20 17:48:52 {plot_text}", color=(255,0,0), position=(11,85), font_size=30)
            else:
                frame = draw_zh_cn(frame, f"2021-10-20 17:48:52 未开放", color=(255, 0, 0), position=(11, 85),
                                   font_size=30)
                frame = draw_zh_cn(frame, f"2021-10-20 17:48:56 {plot_text}", color=(255, 0, 0), position=(11, 115), font_size=30)
        elif frame_id <= 1000:
            frame = draw_zh_cn(frame, f"2021-10-20 17:48:52 未开放", color=(255,0,0), position=(11,85), font_size=30)
            frame = draw_zh_cn(frame, f"2021-10-20 17:48:56 {plot_text}", color=(255, 0, 0), position=(11, 115), font_size=30)
        else:
            frame = draw_zh_cn(frame, f"2021-10-20 17:48:52 未开放", color=(255,0,0), position=(11,85), font_size=30)
            frame = draw_zh_cn(frame, f"2021-10-20 17:48:56 空闲中", color=(255, 0, 0), position=(11, 115), font_size=30)
            frame = draw_zh_cn(frame, f"2021-10-20 17:49:31 {plot_text}", color=(255, 0, 0), position=(11, 145), font_size=30)
        # else:
        #     print("Frame {} no detections".format(frame_id))

        return frame

    @staticmethod
    def print_timing_info():
        log.debug('=================Timing Stats=================')
        log.debug(f"{'detect:':<37}{Profiler.get_avg_millis('detect'):>6.3f} ms")
        log.debug(f"{'mob2staff:':<37}{Profiler.get_avg_millis('mob2staff'):>6.3f} ms")
    # log.debug(f"{'track:':<37}{Profiler.get_avg_millis('track'):>6.3f} ms")


def demo(cfg, video_path):
    video_name = os.path.basename(video_path)
    outputFile = str(cfg.log_dir / f"demo_{video_name}")
    width, height = 1280, 720
    if cfg.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  ## MJPG MP4V
        output = cv2.VideoWriter(outputFile, fourcc, 5, (width, height), )
    fvs = FileVideoStream(video_path).start()
    mot = MOT(cfg)

    frame_id = 0
    frame_count = 0
    while fvs.running():
        if frame_id % (cfg.skip_frame + 1) != 0 or frame_id < 0:
            frame = fvs.read()
            frame_id += 1
            continue
        frame = fvs.read()
        if frame is None: break

        log.info(f'\n=================New Frame {frame_count}=================\n')
        frame = cv2.resize(frame, (width, height))
        frame = mot.step(frame, frame_id, frame_count, video_name)

        mot.print_timing_info()

        frame_count += 1
        frame_id += 1
        if cfg.save_video:
            output.write(frame)

        if cfg.show_predict_video:
            cv2.imshow('test', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # suspend 按s键会暂停
                cv2.waitKey(0)
            if key == ord('c'):  # continue 按c键不动的话，视频会持续运行
                continue
            if key == ord('q'):  # quit
                exit()

    cv2.destroyAllWindows()
    fvs.stop()


def traversal_videos(cfg):
    creat_log_dir(cfg)
    # videos in folder
    video_list = ["/mnt/shy/农行POC/算法技术方案/demo_1021/窗口工作状态/original.mp4"]  # sorted(glob.glob(os.path.join(cfg.video_path, "*.mp4")))
    for i, video_path in enumerate(video_list):
        print(f"{i} / {len(video_list) - 1}")
        print(f"====> {video_path}")
        demo(cfg, video_path)


# folders in folder
# folders = sorted(os.listdir(cfg.video_path))
# video_list = []
# for folder in folders:
# 	current_folder = os.path.join(cfg.video_path, folder)
# 	video_list += glob.glob(os.path.join(current_folder, "*.mp4"))
# for i, video_path in enumerate(video_list):
# 	print(f"{i} / {len(video_list) - 1}")
# 	print(f"====> {video_path}")
# 	demo(cfg, video_path)


def parse_args():
    parser = argparse.ArgumentParser(description="MOT")
    # parser.add_argument(
    # 	"--yml", default="./configs/track.yml")
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--save_img", default=False)
    parser.add_argument("--show_predict_video", default=False)
    parser.add_argument(
        "--video_path",
        # default="/mnt/shy/track/test_yze/cut/guimian_05.mp4"
        # default="/mnt/shy/农行POC/abc_data/第五批0926/cut_video/C26_2_0923_1000_1020_000000--000200.mp4"
        default="/mnt/shy/农行POC/abc_data/第六批1020/Badcase"

        # default="/mnt/shy/track/test_yze/guimian.mp4"
    )
    # person  head_shoulder  face
    parser.add_argument(
        "--track_type",
        # default="head_shoulder"
        default="person"
        # default="bank_person_hs"
    )
    parser.add_argument(
        "--skip_frame", default=5, type=int)
    argument = parser.add_argument("--feature_weight", default=0.9, type=float)
    parser.add_argument(
        "--log_dir",
        # default="/mnt/shy/track/test_yze/logs/"
        default="/mnt/shy/农行POC/算法技术方案/demo_1021/窗口工作状态/draw/"
    )

    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    traversal_videos(cfg)
