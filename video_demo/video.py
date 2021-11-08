import os
import cv2
import time
import datetime
from threading import Thread
from queue import Queue
from ipdb import set_trace as pause

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self.frame_id = 1

	def start(self):
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self.frame_id += 1
		self.stop()

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self.frame_id / self.elapsed()

class FileVideoStream:
	def __init__(self, path, img_size=None, queue_size=256):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		assert os.path.isfile(path), "Error: path error"		
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		self.img_size = img_size

		# initialize the queue used to store frames read from the video file
		self.Q = Queue(maxsize=queue_size)
		# intialize thread
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True

	def start(self):
		# start a thread to read frames from the file video stream
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				break

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stopped = True
					
				# if there are transforms to be done, might as well
				# do them on producer thread before handing back to
				# consumer thread. ie. Usually the producer is so far
				# ahead of consumer that we have time to spare.
				#
				# Python is not parallel but the transform operations
				# are usually OpenCV native so release the GIL.
				#
				# Really just trying to avoid spinning up additional
				# native threads and overheads of additional
				# producer/consumer queues since this one was generally
				# idle grabbing frames.
				if self.img_size:
					frame = cv2.resize(frame, self.img_size)

				# add the frame to the queue
				self.Q.put(frame)
			else:
				time.sleep(0.5)  # Rest for 10ms, we have a full queue

		self.stream.release()

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
	def running(self):
		return self.more() or not self.stopped

	def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
		tries = 0
		while self.Q.qsize() == 0 and not self.stopped and tries < 5:
			time.sleep(0.1)
			tries += 1

		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		self.thread.join()