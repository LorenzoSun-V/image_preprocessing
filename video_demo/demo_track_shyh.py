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
from utils import checkfolder, convert_det_dict, crop_im, plot_one_box, decode_ratio, if_inPoly


label_list3 = ['bank_staff_summer', 'bank_staff_fall', 'custom', 'security_staff_summer', 'security_staff_fall', 'volunteer', 'cleaner']
color_list3 = [(169, 169, 169), (0, 0, 255), (0, 128, 128), (130, 0, 75), (203, 192, 255), (0, 165, 255), (235, 206, 135)]
color_dict3 = {k: v for k, v in zip(label_list3, color_list3)}
roi_dict = {"C28": [(246, 176), (337, 713), (1167, 714), (1218, 383), (794, 99)],
			"C31": [(35, 284), (128, 720), (1260, 720), (1261, 529), (1085, 424), (1112, 345), (543, 161)],
			"C32": [(345, 244), (319, 499), (149, 505), (117, 720), (1159, 720), (805, 387), (888, 367), (670, 232)],
			"C38": [(412, 226), (8, 407), (0, 720), (1200, 720), (1200, 177), (922, 115), (562, 271)],
			"C39": [(583, 317), (742, 720), (974, 720), (1154, 648), (1193, 553), (1029, 411), (748, 425), (689, 319)],
			"C49": [(719, 147), (468, 126), (97, 314), (71, 720), (997, 720), (1237, 179), (1165, 132), (1011, 278)],
			"C56": [(661, 149), (316, 219), (203, 89), (46, 127), (89, 720), (1095, 720), (1213, 460)]}


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
	def __init__(self, cfg, debug=False):
		self.cfg = cfg
		self.det_model = SoftFPN(detect_type='bank_person_hs')
		self.cls_model = SoftFPN(detect_type='mob2shbank7')

		if cfg.track_type == "person":
			self.feat_model = SoftFPN('person_feature')
		elif cfg.track_type == "head_shoulder":
			self.feat_model = SoftFPN('hs_feature')
		else:
			raise TypeError(f" wrong track type : {cfg.track_type}, please check")

	def step(self, frames, frame_id, video_name):
		camera_num = video_name.split("_")[0]
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
					if if_inPoly(roi_dict[camera_num], roi_point) and (h > 35 and w > 35):
						ims.append(crop_im(frames[index], tlbr))
						tlbrs_roi.append(tlbr)
				# ims = [crop_im(frames[index], tlbr) for tlbr in tlbrs]

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
						img_name = "{}_({}_{}).jpg".format(video_name.split(".")[0], frame_id-(16-index), i)
						save_img_path = cfg.log_dir / camera_num / "img" / f"{label_list3.index(label)}-{label}" / img_name
						dir_path = save_img_path.parent
						if not dir_path.exists():
							dir_path.mkdir(parents=True)
						cv2.imwrite(str(save_img_path), ims[i])

				# 绘制检测框信息
				with Profiler("ploy bbox"):
					if self.cfg.save_video:
						for i, (tlbr, label, score) in enumerate(zip(tlbrs_roi, labels, scores)):
							text = label + " " + score[:4]
							# text = label+ " " + str(round(score,2))
							plot_one_box(frames[index], tlbr, text, color=color_dict3[label])
			else:
				print("Frame {} no detections".format(frame_id-(16-index)))

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


def demo(cfg, video_path):
	video_name = os.path.basename(video_path)
	camera_num = video_name.split('_')[0]
	outputFile = str(cfg.log_dir / camera_num / f"demo_{video_name}")
	if not os.path.exists(os.path.dirname(outputFile)):
		os.mkdir(os.path.dirname(outputFile))
	width, height = 1280, 720
	if cfg.save_video:
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')  ## MJPG MP4V
		output = cv2.VideoWriter(outputFile, fourcc, 15, (width, height))
	fvs = FileVideoStream(video_path).start()
	mot = MOT(cfg)

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
				if cfg.save_video:
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


def traversal_videos(cfg):
	creat_log_dir(cfg)
	# videos in folder
	# video_list = sorted(glob.glob(os.path.join(cfg.video_path, "*.mp4")))
	#
	# for i, video_path in enumerate(video_list):
	# 	print(f"{i} / {len(video_list) - 1}")
	# 	print(f"====> {video_path}")
	# 	demo(cfg, video_path)

	# folders in folder
	folders = sorted(os.listdir(cfg.video_path))
	video_list = []
	for folder in folders:
		current_folder = os.path.join(cfg.video_path, folder)
		video_list += glob.glob(os.path.join(current_folder, "*.mp4"))
	video_list = ["/mnt2/sjh/上海银行data/do_demo/C38/C38_1118_1141_1402_000000--001500_000020--000040.mp4"]   #  sorted(video_list)
	for i, video_path in enumerate(video_list):
		print(f"{i} / {len(video_list) - 1}")
		print(f"====> {video_path}")
		demo(cfg, video_path)


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
		default="/mnt1/shy1/上海银行模型验证/第一批badcase/video/"

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
		"--skip_frame", default=15, type=int)
	# argument = parser.add_argument("--feature_weight", default=0.9, type=float)
	parser.add_argument(
		"--log_dir",
		# default="/mnt/shy/track/test_yze/logs/"
		default="/mnt2/sjh/上海银行data/test_folder/"
	)

	return parser.parse_args()


if __name__ == "__main__":
	cfg = parse_args()
	traversal_videos(cfg)
