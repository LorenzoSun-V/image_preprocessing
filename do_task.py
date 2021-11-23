import os, sys, time
import cv2
import glob
import numpy as np
import json
import shutil
import uuid
import copy
import uuid, time, traceback
from RedisClient import RedisClient
from queue import Queue
import time
import requests
import threading

detect_type_to_bbox = {
  "detect_person": "person_bboxes",
}

color_list = [[0,0,255], [255,0,0], [0,255,0], [255, 255, 0]]

def draw_det_result(frame, color, tlbrs, scores, class_name):
  if tlbrs is not None:
    for i in range(len(tlbrs)):
      x1, y1, w, h = tlbrs[i]
      x2 = x1 + w
      y2 = y1 + h
      frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
      if scores is not None:
        text = str(round(scores[i], 2))
        len_text = len(text)
        (txt_w, txt_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        # (label_width2, label_height2), baseline2 = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        # cv2.rectangle(frame, (x1, y1-label_height1), (x1+label_width1+label_width2, y1), (0, 0, 0), thickness=-1)
        
        cv2.putText(frame, class_name, (x1, y1- int(0.2 * txt_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        cv2.putText(frame, text, (x1+txt_w, y1- int(0.2 * txt_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)


class RedisTask(threading.Thread):
  def __init__(self, source, detect_type):
    threading.Thread.__init__(self)
    redis_ip = "192.167.8.144"
    password = "lG5judg6r9S9Wwt4WA-Uid0FzkP8ZkV0XV1kmX941K4"
    self.task_db = RedisClient(redis_ip, port=7379, password=password, db="0")
    self.result_db = RedisClient(redis_ip, port=7379, password=password, db="1")
    self.detect_db = RedisClient(redis_ip, port=7379, password=password, db="2")
    
    self.source = source
    self.detect_type = detect_type
    
    self.kid = self.get_id()
    self.queue_name = "TaskList" + str(self.kid)
    self.task_id = "atlas" + str(self.kid)
    self.status_id = "TaskList%s_status" % self.kid
    self.result_id = "ResultList" + str(self.kid)
    
    self.img_servername = "http://192.167.8.144:6003/"
    
  def get_id(self):
    kid = None
    for i in range(1, 32):
      key = "TaskList%s_status" % i
      val = self.task_db.get_by_name(key)
      if val:
        val = val.decode("utf-8")
        val_dict = json.loads(val)
        
        if val_dict["status"] == 0 or val_dict["status"] == 110:
          kid = i
      else:
        kid = i
        
      if kid:
        break
      
    return kid
    
  def start_task(self):
    # 清空detect数据
    self.detect_db.delete_by_name(self.task_id)
    self.result_db.delete_by_name(self.result_id)
    os.system("rm -rf ../images/*")
    
    rtsp_json = {
      "source": self.source,
      "operation" : "create_task",
      "task_id" : self.task_id,
      "format" : "H264",
      "task_timstamp": 1573538022,
      "detect_fps" : 1,
      "task_purpose" : self.detect_type,
      "is_save_body_img": 0,
      "is_save_face_img": 0,
      # 只会发送一次
      "rtsp_eof": 1,
    }
    self.task_db.insert_to_list_tail(self.queue_name, json.dumps(rtsp_json))
    
  def stop_task(self):
      rtsp_json = {
          "operation" : "delete_task",
          "task_id" : self.task_id,
          }
      self.task_db.insert_to_list_tail(self.queue_name, json.dumps(rtsp_json))
      
        
  def task_status(self):
    val = self.task_db.get_by_name(self.status_id)
    if val:
      val = val.decode("utf-8")
      val_dict = json.loads(val)
      
      # 所有视频流拉完
      if val_dict["status"] == 110:
        return 110
      # 手动停止
      elif val_dict["status"] == 0:
        return 0
      # 正在执行
      elif val_dict["status"] == 100:
        return 100
    # 当前没有任务
    else:
      return "no task"
    
  def parse_detect_data(self, detect_data):
    # 一帧的检测结果
    item = {}
    
    detect_data = detect_data.decode("utf-8")
    detect_data = json.loads(detect_data)
    
    img_path = detect_data["url_path"].split("/", 3)[-1]
    img_name = os.path.basename(img_path)
    
    url_path = os.path.join(self.img_servername, img_path)
    
    # 下载图片
    response = requests.get(url_path)
    # 获取的文本实际上是图片的二进制文本
    img = response.content
    # 将他拷贝到本地文件 w 写  b 二进制  wb代表写入二进制文本
    with open( '../images/%s' % img_name,'wb' ) as f:
      f.write(img)
    
    item["img_path"] = '../images/%s' % img_name
    
    item["frame_id"] = detect_data["frame_id"]
    item["frame_rid"] = detect_data["frame_idx"]
    print("(frame_id, framerid):", item["frame_id"], item["frame_rid"])
    
    item["bboxes"] = {}
    
    box_key = detect_type_to_bbox[detect_type]
    
    bboxes = detect_data[box_key]
    
    for bbox in bboxes:
      box = {}
      class_name = bbox["class_name"]
      
      if class_name not in item["bboxes"]:
        item["bboxes"][class_name] = {"tlbrs": [], "scores": []}
      
      item["bboxes"][class_name]["tlbrs"].append(bbox["rect"])
      item["bboxes"][class_name]["scores"].append(bbox["conf"])
      
    self.result_db.insert_to_list_tail(self.result_id, json.dumps(item))
  
  
  def run(self):
    print("redis task start")
    fids = []
    while True:
      task_status = self.task_status()
      
      detect_data = self.detect_db.get_and_delete_from_list_head(self.task_id)
      
      if detect_data:
        self.parse_detect_data(detect_data)
        
      else:
        if task_status==110:
          # 获取剩余的数据
          # detect_data = self.detect_db.get_and_delete_from_list_head(self.task_id)
      
          # # 取最后一个数据
          # if detect_data:
          #   self.parse_detect_data(detect_data)
            
          #   self.task_db.delete_by_name(self.status_id)
          #   break
          self.result_db.insert_to_list_tail(self.result_id, json.dumps({"end": 1}))
          self.task_db.delete_by_name(self.status_id)
          break
    print("task_status:", task_status)
    print("redis task end")
    

class DetectAPI(threading.Thread):
  def __init__(self, kid, source):
    threading.Thread.__init__(self)
    redis_ip = "192.167.8.144"
    password = "lG5judg6r9S9Wwt4WA-Uid0FzkP8ZkV0XV1kmX941K4"
    self.kid = kid
    self.source = source
    self.result_db = RedisClient(redis_ip, port=7379, password=password, db="1")
    
    self.result_id = "ResultList" + str(self.kid)
    
    
  def get_frame(self, fid):
    cap = cv2.VideoCapture(self.source)
    
    frame_id = 0
    while True: 
      ret, frame = cap.read()
      
      img_path = '../rimages/%s.jpg' % frame_id
      
      if frame_id == fid:
        break
      
      if frame is None:
        break
      
      frame_id += 1
      
    return frame
  
  def run(self):
    print("detect result start")
    os.system("rm -f %s" % "../detect_images/*")
    
    while True:
      detect_data = self.result_db.get_and_delete_from_list_head(self.result_id)
      
      if detect_data:
        detect_data = detect_data.decode("utf-8")
        detect_data = json.loads(detect_data)
        
        if detect_data.get("end"):
          break
        
        img_path = detect_data["img_path"]
        frame = cv2.imread(img_path)
        
        frame_id = detect_data["frame_id"]
        frame_rid = detect_data["frame_rid"]
        # frame = self.get_frame(frame_rid)
        
        # if frame is None:
        #   break
        
        bboxes = detect_data["bboxes"]
        
        i = 0
        for class_name in bboxes:
          color = color_list[i]
          tlbrs = bboxes[class_name]["tlbrs"]
          scores = bboxes[class_name]["scores"]
          draw_det_result(frame, color, tlbrs, scores, class_name) 
          i += 1
          
        output_img = "../detect_images/%s.jpg" % frame_id
        cv2.imwrite(output_img, frame)
        
    print("detect api end")
    
  
             
    
if __name__ == '__main__':
    # source = "rtsp://192.167.200.99:8554/BigFire.mp4"
    # source = "rtsp://192.167.66.97:8554/01.mp4"
    # source = "rtsp://192.167.10.244:8554/01.mp4"
    source = "rtsp://192.167.3.114:8554/a.mp4"
    detect_type = "detect_person"
    
    redis_task = RedisTask(source, detect_type)
    redis_task.start_task()
    
    print("kid:", redis_task.kid)
    
    kid = redis_task.kid
    
    detect_api = DetectAPI(kid, source)
     
    redis_task.start()
    detect_api.start()
