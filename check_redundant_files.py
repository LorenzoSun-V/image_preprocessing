'''
Author: shy
Description: 图片标注质量检查
LastEditTime: 2021-05-18 14:21:17
'''
import os, cv2, random
from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from utils.utils import plot_one_box


def load_voc_xml( path ):
	xml = open( path, "r" )
	tree = ET.parse( xml )
	xml.close()
	return tree


def find_files(folder, suffix):
	files = []
	files.extend( glob(str(folder / "**.{}".format(suffix))) )
	if len(files) == 0:
		files = find_folder_files(folder, suffix)

	print("{} find {} {} files".format( folder, len(files), suffix ) )
	return files


def find_folder_files(folder, suffix):
	sub_dirs = [x for x in folder.iterdir() if x.is_dir()]
	loop = tqdm(sub_dirs, total=len(sub_dirs))
	files = []
	for sub_dir in loop:
		files.extend( glob(str(sub_dir / "**.{}".format(suffix))) )
	print("{} find {} {} files".format( folder, len(files), suffix ) )
	return files


def check_files(src_names, dst_names):
	lost_names = np.setdiff1d(src_names, dst_names)
	if len(lost_names) > 0:
		print(" {} 文件不一致： \n {}".format( len(lost_names), lost_names ) )
	else:
		print("文件一致")
	return lost_names


# 检查 对应的图片和标注文件  是否存在缺失情况
def find_redundant_files(xml_files, img_files):
	xml_names = np.array( [ x.split("Annotations")[1][:-4] for x in xml_files ] )
	img_names = np.array( [ x.split("JPEGImages" )[1][:-4] for x in img_files ] )
	common_names = np.intersect1d(xml_names, img_names) # 两者共同的图片
	print(" 符合条件的 图片和标注文件有 {}".format( len(common_names) ) )
	
	_ = check_files(xml_names, img_names) # xml文件 找不到对应的图片
	_ = check_files(img_names, xml_names) # 图片 找不到对应的xml文件
	
	random.shuffle(common_names)

	xml_dir = xml_files[0].split("Annotations")[0]
	img_dir = img_files[0].split("JPEGImages" )[0]
	img_paths, xml_paths = [], []
	for common_name in common_names:
		xml_paths.append( xml_dir + "Annotations/" + common_name + ".xml" )
		img_paths.append( img_dir + "JPEGImages/"  + common_name + ".jpg" )

	return img_paths, xml_paths


# 检查图片标注情况
def inspect_img(img_paths, xml_paths, label_list, color_list):
	# xml_paths = ["/mnt1/0_各项目标定结果/目标检测/轮椅_拐杖/农业银行/第一批_拐杖补充结果/Annotations/vlc-record-2021-11-03-14h50m26s-rtsp___192.167.10.126-_002959--003114_0295.xml"]
	# img_paths = ["/mnt1/0_各项目标定结果/目标检测/轮椅_拐杖/农业银行/第一批_拐杖补充结果/JPEGImages/vlc-record-2021-11-03-14h50m26s-rtsp___192.167.10.126-_002959--003114_0295.jpg"]
	for img_path, xml_path in zip(img_paths, xml_paths):

		img = cv2.imread( img_path )
		tree = load_voc_xml( xml_path )
		rt = tree.getroot()

		if not rt.findall( "object" ):
			print(xml_path)
			os.remove(img_path)
			os.remove(xml_path)
			continue

		for obj in rt.findall( "object" ):
			name = obj.find( "name" ).text

			if name in label_list:
				bbox = obj.find( "bndbox" )
				xmin = int(float(bbox.find( "xmin" ).text))
				ymin = int(float(bbox.find( "ymin" ).text))
				xmax = int(float(bbox.find( "xmax" ).text))
				ymax = int(float(bbox.find( "ymax" ).text))
				color = color_list[ label_list.index( name ) ]
				# cv2.rectangle( img, (xmin, ymin), (xmax, ymax), color, 2 )
				object_box = (xmin, ymin, xmax, ymax)
				plot_one_box(object_box, img, label=name, color=color, line_thickness=2)

				lms = obj.find( "lm" )
				if lms is not None:
					x1 = int(float(lms.find( "x1" ).text))
					y1 = int(float(lms.find( "y1" ).text))
					x2 = int(float(lms.find( "x2" ).text))
					y2 = int(float(lms.find( "y2" ).text))
					x3 = int(float(lms.find( "x3" ).text))
					y3 = int(float(lms.find( "y3" ).text))
					x4 = int(float(lms.find( "x4" ).text))
					y4 = int(float(lms.find( "y4" ).text))	
					x5 = int(float(lms.find( "x5" ).text))
					y5 = int(float(lms.find( "y5" ).text))	
					points_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5),]
					for point in points_list:
						cv2.circle(img, point, 1, color, 2)

		# cv2.namedWindow( 'det_result', cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('det_result', 640, 480)
		# cv2.imshow("det_result", img)
		# img_name = os.path.basename(img_path)
		# save_path = os.path.join("/mnt1/0_各项目标定结果/目标检测/轮椅_拐杖/农业银行/vis", img_name)
		# cv2.imwrite(save_path, img)
		# print( xml_path )
	
		ch = cv2.waitKey(0) # 按enter键切换下一张图片
		# 按 ese或q键退出显示

		if ch == "d":
			cv2.destroyAllWindows()  # release windows
			os.remove(xml_path)
			os.remove(img_path)
			print("delete {img_path}")
		elif ch == 27 or ch == ord('q') or ch == ord('Q'):
			break


def find_labels(img_paths, xml_paths, label_list, new_dir):

	new_xml_dir = Path( new_dir + "Annotations/" )
	new_img_dir = Path( new_dir + "JPEGImages/" )

	for img_path, xml_path in zip(img_paths, xml_paths):
		tree = load_voc_xml( xml_path )
		rt = tree.getroot()

		for obj in rt.findall( "object" ):
			name = obj.find( "name" ).text
			if name in label_list:
				if not os.path.exists(Path('/').joinpath(new_img_dir, os.path.dirname(img_path).split("/")[-1])):
					os.makedirs(Path('/').joinpath(new_img_dir, os.path.dirname(img_path).split("/")[-1]))
					os.makedirs(Path('/').joinpath(new_xml_dir, os.path.dirname(xml_path).split("/")[-1]))
				new_img_path = Path('/').joinpath(new_img_dir, os.path.dirname(img_path).split("/")[-1], Path(img_path).name)
				new_xml_path = Path('/').joinpath(new_xml_dir, os.path.dirname(xml_path).split("/")[-1], Path(xml_path).name)
				shutil.copyfile(img_path, new_img_path)
				shutil.copyfile(xml_path, new_xml_path)


def check_img(img_paths):
	loop = tqdm(img_paths, total= len(img_paths))

	for img_path in loop:
		im = cv2.imread(img_path)
		if im is None: print("wrong f{img_path}, can't read")


if __name__ == "__main__":
	# root = "/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/农业银行第三批_轮椅拐杖/"
	# root = "/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/农业银行第三批_拐杖轮椅7.23/"
	root = "/mnt1/0_各项目标定结果/目标检测/轮椅_拐杖/农业银行/第一批_拐杖补充结果/"
	# label_list = [ "wheelchair" ]
	label_list = [ "crutch", "wheelchair"]
	# label_list = ["knife", "scissors", "sharpTools", "expandableBaton", "smallGlassBottle", "electricBaton",
	# 			  "plasticBeverageBottle", "plasticBottleWithaNozzle", "electronicEquipment", "battery",
	# 			  "seal", "umbrella"]
	# xml_dir = Path( root + "Annotations/" )
	# img_dir = Path( root + "JPEGImages/" )
	xml_dir = Path( root + "Annotations/" )
	img_dir = Path( root + "JPEGImages/" )

	# color_list = [ (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)] #cv2 bgr
	color_list = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
				  (240, 32, 160), (79, 79, 47), (147, 20, 255), (179, 222, 245), (86, 114, 255), (197, 181, 255)]
	# 获取 针对二级文件夹的图片路径
	xml_files = find_files(xml_dir, "xml")
	img_files = find_files(img_dir, "jpg")
	# 检查 对应的图片和标注文件  是否存在缺失情况

	# img_files = glob(os.path.join("/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/Y型拐杖数据集/JPEGImages/C48_6_0702_1030-1230", "*.jpg"))
	# xml_files = glob(os.path.join("/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/Y型拐杖数据集/Annotations/C48_6_0702_1030-1230", "*.xml"))
	img_paths, xml_paths = find_redundant_files(xml_files, img_files)

		
	# 检查图片标注情况
	inspect_img(img_paths, xml_paths, label_list, color_list)
	# find_labels(img_paths, xml_paths, label_list, new_dir="/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/拐杖数据集/")
