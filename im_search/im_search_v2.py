'''
Author: shy
Description: 去除重复的图片，构建VP树需要的复杂度为O(n logn)，一次搜索只需要O(logn)的复杂度
LastEditTime: 2021-11-09 17:41:28
'''
import os, cv2, glob, time, click, shutil, functools
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from vptree import VPTree
import scipy.fftpack

from im_hash import phash, hamming_distance

###############################
ROOT_DIR = "/mnt2/private_data/abc_8_remove/img_59/2-money_staff/"
GC_DIR = "/mnt2/private_data/gc_imgs/"   # 回收重复的图片文件夹
DEL_IMG  = False   # 对重复的图片是否删除
PARALLEL = False   # 是否并行
HASH_SIZE = 8 # 图片哈希参数
MAX_HAM_DISTANCE = 10 # 搜索相似图片的哈希距离
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPG', 'PNG'])
###############################


def find_dirs(folder):
	dirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(folder/d)]
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

def build_hash_tree(img_paths, parallel=False):

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

	points = list(hashes.keys())
	tree = VPTree(points, hamming_distance)
	print("[INFO] build tree finished ")
	return tree, hashes


def handle_dup_imgs(hashes):
	remove_imgs_num = 0
	for hash_value, img_paths in hashes.items():
		if len(img_paths)>1:
			for idx,img_path in enumerate(img_paths):
				if idx==0: continue
				delete_img(img_path)
				remove_imgs_num += 1
	print("remove {} image".format(remove_imgs_num))


def handle_similar_imgs(tree, hashes):
	remove_imgs_num = 0
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

			remove_imgs_num += len(similar_img_paths)
	print("remove {} image".format(remove_imgs_num))

def search_imgs(img_dir):
	img_paths = find_files(img_dir)
	print(f"[INFO] dir:{img_dir} num: {len(img_paths)}")
	tree, hashes = build_hash_tree(img_paths, PARALLEL)
	# handle_dup_imgs(hashes)
	handle_similar_imgs(tree, hashes)

if __name__ == '__main__':
	img_dirs = find_dirs(ROOT_DIR)
	if len(img_dirs) > 0:
		with ThreadPoolExecutor() as e:
			loop = tqdm(e.map(search_imgs, img_dirs), total=len(img_dirs))
	else:
		search_imgs(ROOT_DIR)