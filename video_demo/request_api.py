import os
import cv2
import requests
import numpy
import json
from queue import Queue
from threading import Thread


class MyThread(Thread):
	def __init__(self, func, args):
		super(MyThread, self).__init__()
		self.func = func
		self.args = args

	def run(self):
		self.result = self.func(*self.args)

	def get_result(self):
		try:
				return self.result
		except Exception:
				return None


# 将图片frame转化为字节流的耗时任务交给多线程来处理
def task(frame):
    img_data = cv2.imencode(".jpg", frame)[1].tobytes()
    return img_data


class SoftFPN():
	def __init__(self, detect_type, debug=False):
		self.detect_type = detect_type
		self.HOST = 'http://192.167.10.244:7777'
		self.login_url = "%s/login" % self.HOST
		self.upload_url = "%s/upload/api/image" % self.HOST
		self.detect_url = "%s/detect/api/img" % self.HOST
		self.detect_type = detect_type
		self.debug = debug

	def __get_sess(self):
		userAgent = "Mozilla/5.0 (Windows NT 6.1; WOW64) \
					 AppleWebKit/537.36 (KHTML, like Gecko) \
					 Chrome/63.0.3239.132 Safari/537.36"
		header = {'User-Agent': userAgent, }
		data   = {'username':'admin','password':'admin123456'}

		# 通过session模拟登录，每次请求带着session
		self.sess = requests.Session()
		f = self.sess.post(self.login_url, data=data, headers=header)
		# print(json.loads(f.text))

	def __call__(self, detect_data, frame_id=None):
		self.__get_sess()

		data = {
			'detect_type': self.detect_type,
		}

		# 如果多张图片的numpy数组，转换为字节流
		if isinstance(detect_data, list):
			self.multiple_files = []

			frame_queue = Queue()

			for frame in detect_data:
				frame_queue.put(frame)

			threads_list = []

			while not frame_queue.empty():
				frame = frame_queue.get()
				t = MyThread(task, args=(frame,))
				threads_list.append(t)
				t.start()

			i = 0
			for mythread in threads_list:
				mythread.join()
				img_data = mythread.get_result()
				img_name = "frame_%s.jpg" % i
				upload_file = ('images', (img_name, img_data, 'image/png'))
				self.multiple_files.append(upload_file)
				i += 1

			# print("multiple_files:", len(self.multiple_files))
			# 上传图片
			resp = self.sess.post(self.upload_url, data=data, files=self.multiple_files)

		# 如果是一张图片的numpy数组数据
		elif isinstance(detect_data, numpy.ndarray):
			frame = detect_data
			# 图片的字节流
			img_data = cv2.imencode(".jpg", frame)[1].tobytes()
			img_name = str(frame_id) + ".jpg"

			files = [('images', (img_name, img_data, 'image/png'))]
			# 上传图片
			resp = self.sess.post(self.upload_url, data=data, files=files)

		# 如果直接是一张图片的地址
		elif isinstance(detect_data, str):
			if os.path.exists(detect_data):
				img_path = detect_data
				img_name = img_path.rsplit("/", 1)[-1]
				img_data = open(img_path, 'rb')
				frame = cv2.imread(img_path)

				# 图片的字节流
				img_data = cv2.imencode(".jpg", frame)[1].tobytes()
				img_name = str(frame_id) + ".jpg"

				files = [('images', (img_name, img_data, 'image/png'))]
				# 上传图片
				resp = self.sess.post(self.upload_url, data=data, files=files)
			else:
				print("The image file does not found!")
				return

		resp = resp.json()

		# print("resp:", resp)
		
		if resp["task_id"]:
			task_id = resp["task_id"]
			result = self.query_result(task_id)

		return result

	def query_result(self, task_id):
		detect_url = "%s/%s" % (self.detect_url, task_id)
		data = {"detect_type": self.detect_type}
		resp = self.sess.post(detect_url, data=data)
		resp = resp.json()["detect_data"]
		return resp