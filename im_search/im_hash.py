'''
Author: shy
Description: 
LastEditTime: 2021-11-08 16:58:26
'''
import cv2, scipy
import numpy as np

def phash(img_path, hash_size=8, highfreq_factor=4):
	"""
	Perceptual Hash computation.
	Implementation follows
	http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
	"""
	img_size = hash_size * highfreq_factor
	image = cv2.imread(img_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (img_size, img_size))

	dct = scipy.fftpack.dct(scipy.fftpack.dct(resized, axis=0), axis=1)
	dctlowfreq = dct[:hash_size, :hash_size]
	med = np.median(dctlowfreq)
	diff = dctlowfreq > med
	hash_value = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
	hash_value = int(np.array(hash_value, dtype="float64"))
	return hash_value


def hamming_distance(a, b):
	# compute and return the Hamming distance between the integers
	return bin(int(a) ^ int(b)).count("1")