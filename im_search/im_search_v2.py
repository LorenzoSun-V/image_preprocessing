'''
Author: shy
Description: 去除重复的图片，构建VP树需要的复杂度为O(n logn)，一次搜索只需要O(logn)的复杂度
LastEditTime: 2021-11-10 10:11:33
'''
import os, cv2, glob, time, click, shutil, functools
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from vptree import VPTree
import scipy.fftpack

from im_hash  import phash, hamming_distance
from profiler import Profiler

###############################
# ROOT_DIR = "/mnt2/private_data/abc_8_remove/img_59/5-bank_staff_shirt/"
ROOT_DIR = "/mnt2/private_data/abc_8_remove/img_59/"
GC_DIR = "/mnt2/private_data/gc_imgs/"   # 回收重复的图片文件夹
DEL_IMG  = False   # 对重复的图片是否删除
PARALLEL = False   # 是否并行
HASH_SIZE = 8 # 图片哈希参数
MAX_HAM_DISTANCE = 6 # 搜索相似图片的哈希距离
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPG', 'PNG'])
###############################


def find_dirs(folder):
	folder = Path(folder)
	dirs = [str(folder/d) for d in sorted(os.listdir(folder)) if os.path.isdir(folder/d)]
	return dirs

def find_files(dir_path, exts):
	files = []
	for ext in exts:
		files.extend( glob.glob(dir_path + "/*." + ext) )
	return files

def hash_img(img_path):
	hash_value = None
	try:
		hash_value = phash(img_path, HASH_SIZE)
	except:
		print(f"[Error] wrong img {img_path}")

	return hash_value


def delete_img(img_path):
	if os.path.isfile(img_path):
		os.remove(img_path)

def move_img(img_path):
	if os.path.isfile(img_path):
		shutil.move(img_path, GC_DIR)

def build_hash(img_paths, parallel=False):
	hashes = {}
	if parallel:
		with ThreadPoolExecutor() as e:
			hash_values = list(tqdm(e.map(hash_img, img_paths), total=len(img_paths)))
			for (hash_value, img_path) in zip(hash_values, img_paths):
				hash_img_paths = hashes.get(hash_value, [])
				hash_img_paths.append(img_path)
				hashes[hash_value] = hash_img_paths
	else:
		loop = tqdm(enumerate(img_paths), total=len(img_paths))
		for (i, img_path) in loop:
			hash_value = hash_img(img_path)
			if hash_value is not None:
				hash_img_paths = hashes.get(hash_value, [])
				hash_img_paths.append(img_path)
				hashes[hash_value] = hash_img_paths

	return hashes

def build_tree(hashes):
	points = list(hashes.keys())
	tree = VPTree(points, hamming_distance)
	print("[INFO] build tree finished ")
	return tree

def handle_dup_imgs(hashes):
	for hash_value, img_paths in hashes.items():
		if len(img_paths)>1:
			for idx,img_path in enumerate(img_paths):
				if idx == 0: continue
				delete_img(img_path)

def handle_similar_imgs(hashes):
	with Profiler('build_tree'):
		tree = build_tree(hashes)

	points = list(hashes.keys())
	for point in tqdm(points):
		results = tree.get_all_in_range(point, MAX_HAM_DISTANCE)
		results = sorted(results)
		for (distance, hash_value) in results:
			if distance==0: continue
			similar_img_paths = hashes.get(hash_value, [])

			if DEL_IMG:
				[delete_img(i) for i in similar_img_paths]
			else:
				[move_img(i) for i in similar_img_paths]


def handle_imgs(img_dir):
	old_img_paths = find_files(img_dir, ALLOWED_EXTENSIONS)
	print(f" [INFO] dir:{img_dir} [OLD] num: {len(old_img_paths)} ")

	with Profiler('build_hash'):
		hashes = build_hash(old_img_paths, PARALLEL)

	# handle_dup_imgs(hashes)
	handle_similar_imgs(hashes)

	new_img_paths = find_files(img_dir, ALLOWED_EXTENSIONS)
	print(f" [INFO] dir:{img_dir} [NEW] num: {len(new_img_paths)} ")
	print(f" [INFO] dir:{img_dir} [REMOVE] num: { len(old_img_paths) - len(new_img_paths) } \n")



if __name__ == '__main__':
	img_dirs = find_dirs(ROOT_DIR)
	if len(img_dirs) > 0:
		with ThreadPoolExecutor() as e:
			loop = tqdm(e.map(handle_imgs, img_dirs), total=len(img_dirs))
	else:
		handle_imgs(ROOT_DIR)

	print('=================Timing Stats=================')
	print(f"{'build_hash:':<37}{Profiler.get_avg_millis('build_hash'):>6.3f} ms")
	print(f"{'build_tree:':<37}{Profiler.get_avg_millis('build_tree'):>6.3f} ms")
