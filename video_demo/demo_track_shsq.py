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
from utils import checkfolder, convert_det_dict, crop_im, plot_one_box


label_list = ['person', 'fall_person', 'crouch_person']
color_list = [(169, 169, 169), (0, 0, 255), (130, 0, 75)]
color_dict = {k: v for k, v in zip(label_list, color_list)}


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
        self.det_model = SoftFPN(detect_type='fall_person_hs')
        self.cls_model = SoftFPN(detect_type='mob2fall_person_h256_w128')

        if cfg.track_type == "person":
            self.feat_model = SoftFPN('person_feature')
        elif cfg.track_type == "head_shoulder":
            self.feat_model = SoftFPN('hs_feature')
        else:
            raise TypeError(f" wrong track type : {cfg.track_type}, please check")

    # self.tracker  = MultiTracker(cfg)

    def step(self, frame, frame_id, video_name):

        # print('detect')
        with Profiler('detect'):
            det_data = self.det_model(frame)
        det_dict = convert_det_dict(det_data)

        if self.cfg.track_type in det_dict.keys():

            # 裁剪目标小图，提取深度特征
            tlbrs = det_dict[cfg.track_type]["tlbrs"]
            ims = [crop_im(frame, tlbr) for tlbr in tlbrs]

            # this is used to crop detect small img
            # if self.cfg.save_img:
            # 	for i, img in enumerate(ims):
            # 		img_name = "{}_({}_{}).jpg".format(video_name[:-4], frame_id, i)
            # 		save_img_path = cfg.log_dir / "img" / img_name
            # 		dir_path = save_img_path.parent
            # 		if not dir_path.exists():
            # 			dir_path.mkdir(parents=True)
            # 		cv2.imwrite(str(save_img_path), img)

            with Profiler('mob2staff'):
                # cls_result = [self.cls_model(im)['class']    for im in ims]
                # labels = [ i['label'] for i in cls_result]
                # scores = [ i['score'] for i in cls_result]
                labels = []
                scores = []
                for i in range(0, len(ims), 16):
                    if (len(ims) - i) < 16:
                        label_dict = self.cls_model(ims[i:])
                    else:
                        label_dict = self.cls_model(ims[i:i+16])
                    labels += label_dict["class"]["label"]
                    scores += label_dict["class"]["score"]

            # 存小图
            if self.cfg.save_img:
                for i, label in enumerate(labels):
                    img_name = "{}_({}_{}).jpg".format(video_name.split(".")[0], frame_id, i)
                    save_img_path = cfg.log_dir / "img" / f"{label_list.index(label)}-{label}" / img_name
                    dir_path = save_img_path.parent
                    if not dir_path.exists():
                        dir_path.mkdir(parents=True)
                    cv2.imwrite(str(save_img_path), ims[i])

            # 绘制检测框信息
            for i, (tlbr, label, score) in enumerate(zip(tlbrs, labels, scores)):
                text = label + " " + score[:4]
                # text = label+ " " + str(round(score,2))
                plot_one_box(frame, tlbr, text, color=color_dict[label])

        else:
            print("Frame {} no detections".format(frame_id))

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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  ## MJPG MP4V
        output = cv2.VideoWriter(outputFile, fourcc, 25, (width, height), )
    fvs = FileVideoStream(video_path).start()
    mot = MOT(cfg)

    frame_id = 0
    frame_count = 0
    while fvs.running():
        if frame_id % (cfg.skip_frame + 1) != 0 or frame_id < 0:
            # frame = fvs.read()
            frame_id += 1
            continue
        frame = fvs.read()
        if frame is None: break

        log.info(f'\n=================New Frame {frame_count}=================\n')
        frame = cv2.resize(frame, (width, height))
        frame = mot.step(frame, frame_count, video_name)
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
    video_list = sorted(glob.glob(os.path.join(cfg.video_path, "*.mp4")))
    video_list = ["/mnt/shy/上海智慧社区探索项目/raw_data_20211027/falldown/cut_1028/ch20_20211026150000_000841--000856.mp4"]
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
        default="/mnt/shy/上海智慧社区探索项目/raw_data_20211022/use"

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
        default="/mnt/shy/sjh/上海智慧社区data/demo/"
    )

    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    traversal_videos(cfg)
