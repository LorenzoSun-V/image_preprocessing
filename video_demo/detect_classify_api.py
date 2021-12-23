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
from utils import checkfolder, convert_det_dict, crop_im, plot_one_box, decode_ratio, if_inPoly, get_all_files


community_shanghai = ['person', 'fall_person', 'crouch_person']
bank_of_shanghai = ['bank_staff_summer', 'bank_staff_fall', 'custom', 'security_staff_summer', 'security_staff_fall',
                    'volunteer', 'cleaner']
ABC = ['bank_staff_vest', 'cleaner', 'money_staff', 'person', 'security_staff', 'bank_staff_shirt',
       'bank_staff_coat', 'security_staff_black']

roi_dict_community_shanghai = None
roi_dict_ABC = None
roi_dict_bank_of_shanghai = {
        "C22": []
        "C28": [(246, 176), (337, 713), (1167, 714), (1218, 383), (794, 99)],
        "C31": [(35, 284), (128, 720), (1260, 720), (1261, 529), (1085, 424), (1112, 345), (543, 161)],
        "C32": [(345, 244), (319, 499), (149, 505), (117, 720), (1159, 720), (805, 387), (888, 367), (670, 232)],
        "C38": [(412, 226), (8, 407), (0, 720), (1200, 720), (1200, 177), (922, 115), (562, 271)],
        "C39": [(583, 317), (742, 720), (974, 720), (1154, 648), (1193, 553), (1029, 411), (748, 425), (689, 319)],
        "C49": [(719, 147), (468, 126), (97, 314), (71, 720), (997, 720), (1237, 179), (1165, 132), (1011, 278)],
        "C56": [(661, 149), (316, 219), (203, 89), (46, 127), (89, 720), (1095, 720), (1213, 460)],
        "C83": [(157, 93), (84, 674), (1040, 688), (960, 75)],
        "C84": [(204, 134), (143, 675), (1069, 687), (962, 80)]}

palette = [[169, 169, 169], [0, 0, 255], [0, 128, 128], [130, 0, 75], [203, 192, 255], [0, 165, 255],
           [235, 206, 135], [90, 90, 90], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255,  0,  0], [0,  0, 142], [0,  0, 70], [0, 60, 100], [0, 80, 100], [0,  0, 230], [119, 11, 32]]


def creat_log_dir(cfg):
    prefix = time.strftime('%Y-%m%d-%H', time.localtime(time.time()))
    suffix = time.strftime('%Y-%m%d-%H%M%S', time.localtime(time.time())).split('-')[-1]
    log_dir = Path(cfg.log_dir) / prefix
    checkfolder(log_dir)
    log_path = str(log_dir / f"track_log_{suffix}.txt")
    logzero.logfile(log_path, maxBytes=1e6, backupCount=3)
    cfg.log_dir = log_dir

    log.info('\n=================New Log=================\n')
    log.info(cfg)
    pprint(cfg)
    log.info('\n=================New Log=================\n')


class MOT:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.do_detect:
            if cfg.task_name == "bank_of_shanghai" or cfg.task_name == "ABC":
                detect_type = 'bank_person_hs'
            elif cfg.task_name == "community_shanghai":
                detect_type = 'fall_person_hs'
            else:
                raise RuntimeError(f'{cfg.task_name} is not supported!')
            self.det_model = SoftFPN(detect_type=detect_type)
        if cfg.do_classify:
            if cfg.task_name == "bank_of_shanghai":
                class_type = 'mob2shbank7'
            elif cfg.task_name == "ABC":
                class_type = 'mob2staff8'
            elif cfg.task_name == "community_shanghai":
                class_type = 'mob2fall_person_h256_w128'
            else:
                raise RuntimeError(f'{cfg.task_name} is not supported!')
            self.cls_model = SoftFPN(detect_type=class_type)

    def step(self, frames, frame_id, file_name):
        camera_num = file_name.split("_")[0]
        if camera_num == file_name:
            camera_num = "temp"
            global roi_dict
            roi_dict = None
        if self.cfg.do_detect:
            with Profiler('detect'):
                det_datas = self.det_model(frames)

                dec_dict_list = []
                for det_data in det_datas:
                    det_dict = convert_det_dict(det_data)
                    dec_dict_list.append(det_dict)

            for index, det_dict in enumerate(dec_dict_list):
                if self.cfg.track_type in det_dict.keys():
                    # 裁剪目标小图，提取深度特征
                    tlbrs = det_dict[cfg.track_type]["tlbrs"]
                    tlbrs_roi = []
                    # tlbrs = decode_ratio(tlbrs, frame.shape[0], frame.shape[1])
                    # 排除区域外 / 过小的 人体bbox
                    ims = []
                    for tlbr in tlbrs:
                        xmin, ymin, xmax, ymax = tlbr
                        h, w = ymax - ymin, xmax - xmin
                        roi_point = int(xmin + (xmax - xmin) / 2), int(ymax)
                        if roi_dict is not None:
                            if if_inPoly(roi_dict[camera_num], roi_point) and (h > 35 and w > 35):
                                ims.append(crop_im(frames[index], tlbr))
                                tlbrs_roi.append(tlbr)
                        else:
                            if h > 35 and w > 35:
                                ims.append(crop_im(frames[index], tlbr))
                                tlbrs_roi.append(tlbr)
                    if self.cfg.do_classify:
                        with Profiler('mob2staff'):
                            labels = []
                            scores = []
                            for i in range(0, len(ims), 16):
                                if (len(ims) - i) < 16:
                                    label_dict = self.cls_model(ims[i:])[0]
                                else:
                                    label_dict = self.cls_model(ims[i:i + 16])[0]
                                labels += label_dict["class"]["label"]
                                scores += label_dict["class"]["score"]

                        # 存小图
                        if self.cfg.save_img:
                            for i, label in enumerate(labels):
                                img_name = "{}_({}_{}).jpg".format(file_name.split(".")[0], frame_id - (16 - index), i)
                                save_img_path = cfg.log_dir / camera_num / "img" / f"{label_list.index(label)}-{label}" / img_name
                                dir_path = save_img_path.parent
                                if not dir_path.exists():
                                    dir_path.mkdir(parents=True)
                                cv2.imwrite(str(save_img_path), ims[i])

                        # 绘制检测框信息
                        with Profiler("ploy bbox"):
                            if self.cfg.save_results:
                                for i, (tlbr, label, score) in enumerate(zip(tlbrs_roi, labels, scores)):
                                    text = label + " " + score[:4]
                                    # text = label+ " " + str(round(score,2))
                                    plot_one_box(frames[index], tlbr, text, color=color_dict[label])
                else:
                    print("Frame {} no detections".format(frame_id - (16 - index)))

            if roi_dict is not None:
                with Profiler("ploy lines"):
                    for each_frame in frames:
                        cv2.polylines(each_frame, np.array([roi_dict[camera_num]], dtype=np.int32), True, thickness=3,
                                      color=(0, 255, 255))  # 绘制多边形

        return frames

    @staticmethod
    def print_timing_info():
        log.debug('=================Timing Stats=================')
        log.debug(f"{'detect:':<37}{Profiler.get_avg_millis('detect'):>6.3f} ms")
        log.debug(f"{'mob2staff:':<37}{Profiler.get_avg_millis('mob2staff'):>6.3f} ms")
        log.debug(f"{'ploy lines:':<37}{Profiler.get_avg_millis('ploy lines'):>6.3f} ms")
        log.debug(f"{'ploy bbox:':<37}{Profiler.get_avg_millis('ploy bbox'):>6.3f} ms")


def video_demo(cfg, video_path):
    video_name = os.path.basename(video_path)
    camera_num = video_name.split('_')[0]
    outputFile = str(cfg.log_dir / camera_num / f"demo_{video_name}")
    if not os.path.exists(os.path.dirname(outputFile)):
        os.mkdir(os.path.dirname(outputFile))
    width, height = 1280, 720
    if cfg.save_results:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  ## MJPG MP4V
        output = cv2.VideoWriter(outputFile, fourcc, 15, (width, height))
    fvs = FileVideoStream(video_path).start()

    frame_id = 0
    frame_count = 0
    frames = []
    while fvs.running():
        if ((frame_id % (cfg.skip_frame)) != 0) or (frame_id < 0):
            # frame = fvs.read()
            frame_id += 1
            continue
        frame = fvs.read()
        if frame is None:
            break
        frames.append(frame)
        frame_count += 1
        frame_id += 1
        if frame_count % 16 == 0:
            log.info(f'\n=================New Frame {frame_count}=================\n')
            frames = mot.step(frames, frame_count, video_name)
            mot.print_timing_info()

            with Profiler("save videos"):
                if cfg.save_results:
                    for frame in frames:
                        frame = cv2.resize(frame, (width, height))
                        output.write(frame)
            log.debug(f"{'save videos:':<37}{Profiler.get_avg_millis('save videos'):>6.3f} ms")
            if cfg.show_predict_video:
                for frame in frames:
                    cv2.imshow('test', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):  # suspend 按s键会暂停
                        cv2.waitKey(0)
                    if key == ord('c'):  # continue 按c键不动的话，视频会持续运行
                        continue
                    if key == ord('q'):  # quit
                        exit()
            frames = []
    cv2.destroyAllWindows()
    fvs.stop()


def img_demo(cfg, img_list):
    imgs = []
    img_paths = []
    hw_list = []
    img_count = 0
    final_cur = img_count
    for i, img_path in enumerate(img_list):
        log.info(f"{i + 1} / {len(img_list)}")
        img = cv2.imread(img_path)
        w, h, c= img.shape
        hw_list.append((h, w))
        img = cv2.resize(img, (1280, 720))
        imgs.append(img)
        img_paths.append(img_path)
        img_count += 1

        if len(img_list) - final_cur < 16:
            if img_count == len(img_list):
                frames = mot.step(imgs, img_count, "temp")
                if cfg.save_results:
                    for frame, current_img_path, hw in zip(frames, img_paths, hw_list):
                        frame = cv2.resize(frame, hw)
                        img_name = os.path.basename(current_img_path)
                        save_path = os.path.join(cfg.log_dir, img_name)
                        cv2.imwrite(save_path, frame)
                break
        else:
            if (img_count % 16 == 0):
                frames = mot.step(imgs, img_count, "temp")
                if cfg.save_results:
                    for frame, current_img_path, hw in zip(frames, img_paths, hw_list):
                        frame = cv2.resize(frame, hw)
                        img_name = os.path.basename(current_img_path)
                        save_path = os.path.join(cfg.log_dir, img_name)
                        cv2.imwrite(save_path, frame)
                final_cur = img_count
                imgs = []
                img_paths = []
                hw_list = []


def traversal_all_files(cfg):
    global mot
    mot = MOT(cfg)
    creat_log_dir(cfg)
    video_list, img_list = get_all_files(cfg.root_path)
    video_list = ["/mnt/shy/sjh/01.mp4"] # sorted(video_list)
    img_list = sorted(img_list)
    log.info("=> video start")
    for i, video_path in enumerate(video_list):
        log.info(f"{i+1} / {len(video_list)}")
        log.info(f"current video path: {video_path}")
        video_demo(cfg, video_path)

    log.info("=> img start")
    img_demo(cfg, img_list)


def parse_args():
    parser = argparse.ArgumentParser(description="API")
    # task_name: community_shanghai, bank_of_shanghai, ABC
    parser.add_argument("--task_name", type=str, default="ABC")
    parser.add_argument("--do_detect", type=bool, default=True)
    parser.add_argument("--do_classify", type=bool, default=True)
    # save detect and classify results
    parser.add_argument("--save_results", default=True)
    # save sub images from original frames
    parser.add_argument("--save_img", default=False)
    parser.add_argument("--show_predict_video", default=False)
    parser.add_argument("--root_path", default="/mnt2/sjh/农行data/第九批1119/badcase_cut/")
    # /mnt/shy/农行POC/abc_data/第九批1123  /mnt/shy/上海智慧社区探索项目/fall_person_from_web_11_30

    # track_type: person  head_shoulder  face
    parser.add_argument("--track_type", default="person")
    # only useful when input type is video
    parser.add_argument("--skip_frame", default=15, type=int)
    parser.add_argument("--log_dir", default="/mnt/shy/sjh/对比视频/")

    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    global color_dict
    global roi_dict
    global label_list
    label_list = eval(f"{cfg.task_name}")
    roi_dict = eval(f"roi_dict_{cfg.task_name}")
    color_dict = {k: v for k, v in zip(label_list, palette)}
    traversal_all_files(cfg)
