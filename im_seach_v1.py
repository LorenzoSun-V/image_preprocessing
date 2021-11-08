'''
Author: shy
Description: 比较慢的图片搜索，O(n^2)
LastEditTime: 2021-11-08 10:54:27
'''

from PIL import Image
import imagehash
from ipdb import set_trace as pause
import os
from tqdm import tqdm
import threading


def is_image(filename):
	f = filename.lower()
	return f.endswith(".png") or f.endswith(".jpg") or \
		f.endswith(".jpeg") or f.endswith(".bmp") or f.endswith(".gif") or '.jpg' in f

def remove_duplicate_images(img_dir):
	before_image_filenames = [os.path.join(img_dir, path) for path in os.listdir(img_dir) if is_image(path)]

	print("hashing")
	images = {}
	for img_file in sorted(before_image_filenames):
		try:
			hash = HASH_FUNC(Image.open(img_file))
		except Exception as e:
			print('Problem:', e, 'with', img_file)
		# if hash in images:
		#     print(img, '  already exists as', ' '.join(images[hash]))
		#     if 'dupPictures' in img: print('rm -v', img)
		images[hash] = images.get(hash, []) + [img_file]

	print("remove duplicate_images")
	for k, img_list in images.items():
		if len(img_list) > 1:
			# print(" ".join(img_list))
			for i,img_path in enumerate(img_list):
				if i>0: os.remove(img_path)

	after_image_filenames = [os.path.join(img_dir, path) for path in os.listdir(img_dir) if is_image(path)]
	# print("image nums: {}".format(len(image_filenames)))
	print("{} remove image nums: {}-{}={}".format(img_dir,
				len(before_image_filenames), len(after_image_filenames),
				len(before_image_filenames) - len(after_image_filenames)))


def remove_similar_images(img_dir):
	before_image_filenames = [os.path.join(img_dir, path) for path in os.listdir(img_dir) if is_image(path)]

	print("hashing")
	images_hash = {}
	for img_file in sorted(before_image_filenames):
		try:
			hash_value = HASH_FUNC(Image.open(img_file))
		except Exception as e:
			print('Problem:', e, 'with', img_file)
		images_hash[img_file] = hash_value

	print("find similar images")
	similar_images = {}
	for img_file, hash_value in images_hash.items():
		for f, h in images_hash.items():
			if not img_file == f:
				diff = abs(hash_value - h)
				if diff<=HASH_DISTANCE:
					similar_images[img_file] = similar_images.get(img_file, []) + [f]

	print("remove similar images")
	for img_path, img_list in similar_images.items():
		if os.path.isfile(img_path):
			for img in img_list:
				if os.path.isfile(img): os.remove(img)

	after_image_filenames = [os.path.join(img_dir, path) for path in os.listdir(img_dir) if is_image(path)]
	print("{} remove image nums: {}-{}={}".format(img_dir,
				len(before_image_filenames), len(after_image_filenames),
				len(before_image_filenames) - len(after_image_filenames)))


if __name__ == '__main__':
	HASH_FUNC = imagehash.phash
	HASH_DISTANCE = 6   # 哈希距离参数，用于找到距离为6以内的相似图片

	# dir_path = "/mnt2/private_data/abc_8_remove_2/img_59/"
	dir_path = "/mnt2/private_data/abc_8_remove_2/img_50/"
	for idx, target in tqdm(enumerate(os.listdir(dir_path)), total=len(os.listdir(dir_path))):
		d = os.path.join(dir_path, target)
		if os.path.isdir(d):
			print(d)
			# t = threading.Thread(target=remove_duplicate_images, args=(d,))  # 删除重复的图片
			t = threading.Thread(target=remove_similar_images, args=(d,))   # 删除相似的图片
			t.start()