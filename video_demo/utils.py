'''
Author: shy
Description: 
LastEditTime: 2021-09-27 10:59:50
'''
import os,cv2
from pathlib import Path, PosixPath
import numpy as np


def checkfolder(paths):
	if isinstance(paths, str):
		if not Path(paths).is_dir():
			os.mkdir(paths)
			print("Created new directory in %s" % paths)

	if isinstance(paths, PosixPath):
		if not Path(paths).is_dir():
			Path.mkdir(paths)
			print("Created new directory in %s" % paths)

def convert_det_dict(det_data):
	det_dict = {}
	for data in det_data:
		class_name, tlbrs, scores = data
		det_dict[class_name] = {}
		det_dict[class_name]["tlbrs"] = tlbrs
		det_dict[class_name]["scores"] = scores
	return det_dict

def plot_one_box(img, tlbr, info=False, color=(0, 255, 0), line_thickness=2):
	tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
	# color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl)
	if info:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(info, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1)  # filled
		cv2.putText(img, info, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def crop_im(frame, tlbr):
	tlbr = [max(x, 0) for x in tlbr]
	# print("tlbr:", tlbr)
	x0, y0, x1, y1 = tlbr
	roi = frame[y0:y1, x0:x1]
	return roi

def decode_ratio(tlbrs, height, width):
  tlbrs = np.array(tlbrs)
  # print("tlbr shape", tlbrs.shape)
  tlbrs[:, 0] *=  width
  tlbrs[:, 1] *=  height
  tlbrs[:, 2] *=  width
  tlbrs[:, 3] *=  height
  return tlbrs.astype("int")