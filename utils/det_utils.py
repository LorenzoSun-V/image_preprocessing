'''
Author: shy
Description: 用于检测器的一些常用函数, 如 iou nms
LastEditTime: 2021-03-22 11:03:20
'''
import os, yaml, cv2
import json, base64
from easydict import EasyDict as edict
import numpy as np
from ipdb import set_trace as pause
import torch

def encode_ratio(tlbrs, height, width):
  tlbrs = np.array(tlbrs)
  tlbrs = tlbrs.astype("float")

  tlbrs[:, 0] /=  width
  tlbrs[:, 1] /=  height
  tlbrs[:, 2] /=  width
  tlbrs[:, 3] /=  height
  return tlbrs

def encode_landmark_ratio(tlbrs, pts, height, width):
    tlbrs = np.array(tlbrs)
    tlbrs = tlbrs.astype("float")
    tlbrs[:, 0] /= width
    tlbrs[:, 1] /= height
    tlbrs[:, 2] /= width
    tlbrs[:, 3] /= height

    pts = np.array(pts)
    pts = pts.astype("float")
    pts[:, 0] /= width
    pts[:, 1] /= height
    pts[:, 2] /= width
    pts[:, 3] /= height
    pts[:, 4] /= width
    pts[:, 5] /= height
    pts[:, 6] /= width
    pts[:, 7] /= height
    pts[:, 8] /= width
    pts[:, 9] /= height

    return tlbrs, pts

def decode_ratio(tlbrs, height, width):
  tlbrs = np.array(tlbrs)
  # print("tlbr shape", tlbrs.shape)
  tlbrs[:, 0] *=  width
  tlbrs[:, 1] *=  height
  tlbrs[:, 2] *=  width
  tlbrs[:, 3] *=  height
  return tlbrs.astype("int")

def decode_landmark_ratio(tlbrs, pts, height, width):
  tlbrs = np.array(tlbrs)
  tlbrs[:, 0] *= width
  tlbrs[:, 1] *= height
  tlbrs[:, 2] *= width
  tlbrs[:, 3] *= height

  pts = np.array(pts)
  pts[:, 0] *= width
  pts[:, 1] *= height
  pts[:, 2] *= width
  pts[:, 3] *= height
  pts[:, 4] *= width
  pts[:, 5] *= height
  pts[:, 6] *= width
  pts[:, 7] *= height
  pts[:, 8] *= width
  pts[:, 9] *= height

  return tlbrs.astype("int"), pts.astype("int")

def draw_det_result(frame, color, tlbrs, scores, class_name):
  if tlbrs is not None:
    for i in range(len(tlbrs)):
      x1, y1, x2, y2 = tlbrs[i]
      frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
      if scores is not None:
        text = str(round(scores[i], 2))
        len_text = len(text)
        (txt_w, txt_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        # (label_width2, label_height2), baseline2 = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        # cv2.rectangle(frame, (x1, y1-label_height1), (x1+label_width1+label_width2, y1), (0, 0, 0), thickness=-1)
        
        cv2.putText(frame, class_name, (x1, y1- int(0.2 * txt_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        cv2.putText(frame, text, (x1+txt_w, y1- int(0.2 * txt_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        
colors = {
    "red": (255, 0, 0),  # Red
    "green": (0, 255, 0),  # Green
    "blue": (0, 0, 255),  # Blue
    # "yellow":     (255, 255, 0),  # Yellow
    "magenta": (255, 0, 255),  # Magenta 红紫色 洋红色
    "cyan": (0, 255, 255),  # Cyan 青色
    "purple": (155, 48, 255),
    # "light gray": (192, 192, 192),  # Light gray
    # "dark gray":  (64, 64, 64),  # Dark gray
    # "black":      (0, 0, 0),  # Black
    # "white":      (255, 255, 255),  # White
}
     
def draw_landmark_det_result(frame, color, tlbrs, scores, pts, class_name):
  if tlbrs is not None:
    for i in range(len(tlbrs)):
          tlbr = tlbrs[i]
          score = scores[i]
          pt = pts[i]
          # info = class_name + f" {round(score,2)}"
          info = class_name + " {}".format(round(score, 2))
          plot_one_box(frame, tlbr, pt, info, color)
  return frame
   
def plot_one_box(img, tlbr, pt, info, color, line_thickness=2):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3]))
    pt0, pt1, pt2, pt3, pt4 = (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (int(pt[4]), int(pt[5])), (
    int(pt[6]), int(pt[7])), (int(pt[8]), int(pt[9]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    cv2.circle(img, pt0, 1, color, 2)
    cv2.circle(img, pt1, 1, color, 2)
    cv2.circle(img, pt2, 1, color, 2)
    cv2.circle(img, pt3, 1, color, 2)
    cv2.circle(img, pt4, 1, color, 2)

    if info:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(info, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, info, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)  
        
                
def softfpn_preprocess_img(img, height, width):
  img_resize = cv2.resize(img, ( width, height ))  
  img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
  img_data = img_rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255
  img_data= np.ascontiguousarray(img_data)
  return img_data


def exfeature_preprocess_img(img, height, width):
    im = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    im = im.astype(np.float32, copy=False)
    im = im / 255.0
    im -=[[[0.485, 0.456, 0.406]]]
    im /=[[[0.229, 0.224, 0.225]]]
    im = im.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    np_im= np.ascontiguousarray(im)
    return np_im


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, np.int64):
      return int(obj)
    if isinstance(obj, np.float32):
      return float(obj)
    if isinstance(obj, bytes):
      return str(obj, encoding='utf-8')
    return json.JSONEncoder.default(self, obj)

def base64_encode_img_arr(a):
  return base64.b64encode(a).decode("utf-8")

def base64_decode_img_arr(a, shape):
  a = bytes(a, encoding="utf-8")
  # for process image
  # a = np.frombuffer(base64.decodestring(a), dtype=np.float32) 
  a = np.frombuffer(base64.decodebytes(a), dtype=np.uint8)
  # print(a)
  a = a.reshape(shape)
  return a

def read_yml(yml_file):
  with open(yml_file) as f:
    cfg = edict(yaml.safe_load(f))
  return cfg

def get_cfg(detect_type):
  dir_path = os.path.dirname(os.path.abspath(__file__)) + "/config"
  yml_file = "%s/%s.yaml" % (dir_path, detect_type)
  cfg = read_yml(yml_file)

def get_iou(A, B):
  """ iou(A,B) = |A∩B| / |A∪B|
  
  A: list, tlbr [x0, y0 , x1, y1]
  B: list, tlbr [x0, y0 , x1, y1]
  """
  # 交集box的tlbr坐标
  max_x0 = np.maximum( A[0], B[0] ) 
  max_y0 = np.maximum( A[1], B[1] ) 
  min_x1 = np.minimum( A[2], B[2] )
  min_y1 = np.minimum( A[3], B[3] )
  
  # 计算 交集box的面积
  w = np.maximum( 0., min_x1 - max_x0 )
  h = np.maximum( 0., min_y1 - max_y0 )
  area_inter = w * h

  # 计算 并集面积
  area_A = ( A[2]-A[0] ) * ( A[3]-A[1] ) # w*h
  area_B = ( B[2]-B[0] ) * ( B[3]-B[1] )
  area_union = area_A + area_B - area_inter
  
  iou = area_inter / area_union
  return iou

def get_dice(A, B):
  """ dice(A,B) = 2*|A∩B| / (|A|+|B|)

  A: list, tlbr [x0, y0 , x1, y1]
  B: list, tlbr [x0, y0 , x1, y1]
  """
  # 交集box的tlbr坐标
  max_x0 = np.maximum( A[0], B[0] ) 
  max_y0 = np.maximum( A[1], B[1] ) 
  min_x1 = np.minimum( A[2], B[2] )
  min_y1 = np.minimum( A[3], B[3] )
  
  # 计算 交集box的面积
  w = np.maximum( 0., min_x1 - max_x0 )
  h = np.maximum( 0., min_y1 - max_y0 )
  area_inter = w * h

  area_A = ( A[2]-A[0] ) * ( A[3]-A[1] ) # w*h
  area_B = ( B[2]-B[0] ) * ( B[3]-B[1] )
  
  dice = 2 * area_inter / ( area_A + area_B )
  return dice


def nms(results, min_score, max_iou):
  """ 如果多个box重叠度很高，只保留分数最高的那一个，其余的忽略

  1. 将box_list 按置信度分数进行排序
  2. 选择分数最高的box, 与 box_list 当中其他所有框计算重叠度iou
    - 小于阈值的box保留, 大于阈值的 box 从 box_list 中移除
    - 将当前的box加入到 keep_list 当中
  3. 从 box_list 中继续重复第二步, 直到 box_list被完全遍历
  4. 返回 keep_list
  """
  if len(results) == 0: 
    return [], []

  keep_list = []
  results = np.array(results)
  tlbrs =  results[:, 0:4]
  scores = results[:,4]
  
  # 按置信度从大到小进行排序, 返回index
  score_order = scores.argsort()[::-1]
  remove_list = []
  while len(score_order) > 0:
    max_score_index = score_order[0]
    box = tlbrs[max_score_index]

    keep_inds = []
    for ( i, index ) in enumerate( score_order[1:] ):
      iou = get_iou(box, tlbrs[index])      
      if( iou <= max_iou ):
        keep_inds.append( i+1 ) 
    score_order = score_order[keep_inds]

    keep_list.append(max_score_index)

  return tlbrs[keep_list].astype("int"), scores[keep_list]


def robust_nms(dets, min_score, max_dice):
  """ 加权平均 nms : 
      如果多个box重叠度很高，就把它们通过加权平均的方式融合成一个
      定位更加精准, 但无法改善遮挡情况
  
  1. 将box_list 按置信度分数进行排序
  2. 为每个box分配一个标签，相互重叠度很高的box会得到相同的标签
  3. 找出所有相同标签的box，并计算加权平均
  4. 用加权平均值代替原来的坐标
  """
  if len(dets) == 0: 
    return [], []

  keep_list = []
  keep_dets = []
  dets = np.array(dets)
  tlbrs =  dets[:, 0:4]
  scores = dets[:,4]
  # 按置信度从大到小进行排序, 返回index
  score_order = scores.argsort()[::-1]
  
  # 相互重叠度很高的box放在一个list里面
  label_list = []
  while len(score_order) > 0:
    max_score_index = score_order[0]
    box = tlbrs[max_score_index]

    keep_inds = []
    same_inds = [max_score_index]
    for ( i, index ) in enumerate( score_order[1:] ):
      iou = get_dice(box, tlbrs[index])
      if( iou <= max_dice ):
        keep_inds.append( i+1 )
      else:
        same_inds.append( index )
    
    score_order = score_order[keep_inds]
    keep_list.append(max_score_index)
    label_list.append(same_inds)

  # 计算加权平均值, 用加权平均值代替原来的坐标
  nms_scores = scores[keep_list]
  nms_tlbrs  = tlbrs[keep_list]
  for ( i, tlbr ) in enumerate( nms_tlbrs ):

    sum_x0 = 0.0
    sum_y0 = 0.0
    sum_x1 = 0.0
    sum_y1 = 0.0
    sum_w  = 0.0

    box_inds = label_list[i]
    for j in box_inds:
      w = scores[j] - min_score + 0.001
      sum_x0 += w * tlbrs[j][0]
      sum_y0 += w * tlbrs[j][1]
      sum_x1 += w * tlbrs[j][2]
      sum_y1 += w * tlbrs[j][3]
      sum_w  += w

    tlbr[0] = sum_x0 / sum_w
    tlbr[1] = sum_y0 / sum_w
    tlbr[2] = sum_x1 / sum_w
    tlbr[3] = sum_y1 / sum_w

  return nms_tlbrs, nms_scores


def landmark_robust_nms(dets, min_score, max_dice):
    """ 加权平均 nms :
            如果多个box重叠度很高，就把它们通过加权平均的方式融合成一个
            定位更加精准, 但无法改善遮挡情况

    1. 将box_list 按置信度分数进行排序
    2. 为每个box分配一个标签，相互重叠度很高的box会得到相同的标签
    3. 找出所有相同标签的box，并计算加权平均
    4. 用加权平均值代替原来的坐标
    """
    if len(dets) == 0:
        return [], []

    keep_list = []
    keep_dets = []
    dets = np.array(dets)
    tlbrs = dets[:, 0:4]
    scores = dets[:, 4]
    # add 5 face keypoints
    pts = dets[:, 5:15]

    # 按置信度从大到小进行排序, 返回index
    score_order = scores.argsort()[::-1]

    # 相互重叠度很高的box放在一个list里面
    label_list = []
    while len(score_order) > 0:
        max_score_index = score_order[0]
        box = tlbrs[max_score_index]

        keep_inds = []
        same_inds = [max_score_index]
        for (i, index) in enumerate(score_order[1:]):
            iou = get_dice(box, tlbrs[index])
            if (iou <= max_dice):
                keep_inds.append(i + 1)
            else:
                same_inds.append(index)

        score_order = score_order[keep_inds]
        keep_list.append(max_score_index)
        label_list.append(same_inds)

    # 计算加权平均值, 用加权平均值代替原来的坐标
    nms_scores = scores[keep_list]
    nms_tlbrs = tlbrs[keep_list]
    # add 5 face keypoints
    nms_pts = pts[keep_list]
    # print('======= ', len(nms_scores), len(nms_tlbrs), len(nms_pts), keep_list, nms_pts)
    for (i, tlbr) in enumerate(nms_tlbrs):
        sum_x0 = 0.0
        sum_y0 = 0.0
        sum_x1 = 0.0
        sum_y1 = 0.0
        sum_w = 0.0
        box_inds = label_list[i]
        for j in box_inds:
            w = scores[j] - min_score + 0.001
            sum_x0 += w * tlbrs[j][0]
            sum_y0 += w * tlbrs[j][1]
            sum_x1 += w * tlbrs[j][2]
            sum_y1 += w * tlbrs[j][3]
            sum_w += w
        tlbr[0] = sum_x0 / sum_w
        tlbr[1] = sum_y0 / sum_w
        tlbr[2] = sum_x1 / sum_w
        tlbr[3] = sum_y1 / sum_w

    # add 5 face keypoints
    for (i, kpt) in enumerate(nms_pts):
        sum_x0 = 0.0
        sum_y0 = 0.0
        sum_x1 = 0.0
        sum_y1 = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        sum_x3 = 0.0
        sum_y3 = 0.0
        sum_x4 = 0.0
        sum_y4 = 0.0
        sum_w = 0.0
        box_inds = label_list[i]
        for j in box_inds:
            w = scores[j] - min_score + 0.001
            sum_x0 += w * pts[j][0]
            sum_y0 += w * pts[j][1]
            sum_x1 += w * pts[j][2]
            sum_y1 += w * pts[j][3]
            sum_x2 += w * pts[j][4]
            sum_y2 += w * pts[j][5]
            sum_x3 += w * pts[j][6]
            sum_y3 += w * pts[j][7]
            sum_x4 += w * pts[j][8]
            sum_y4 += w * pts[j][9]
            sum_w += w
        kpt[0] = sum_x0 / sum_w
        kpt[1] = sum_y0 / sum_w
        kpt[2] = sum_x1 / sum_w
        kpt[3] = sum_y1 / sum_w
        kpt[4] = sum_x2 / sum_w
        kpt[5] = sum_y2 / sum_w
        kpt[6] = sum_x3 / sum_w
        kpt[7] = sum_y3 / sum_w
        kpt[8] = sum_x4 / sum_w
        kpt[9] = sum_y4 / sum_w

    return nms_tlbrs, nms_scores, nms_pts
  
  
def soft_nms(results, min_score, max_dice, method="linear"):
  """ 衰减nms : 如果多个box重叠度很高，根据重叠度降低相邻框的置信度, 最后设置阈值删除
    可改善密集目标 遮挡检测结果
  
  1. 将box_list 按置信度分数进行排序
  2. 选择分数最高的box, 与 box_list 当中其他所有框计算重叠度iou
    - 小于阈值的box保留, 大于阈值的 box 根据重叠度降低对应置信度
    - 将当前的box加入到 keep_list 当中
  3. 从 box_list 中继续重复第二步, 直到 box_list被完全遍历
  4. 将所有的box按置信度分值进行过滤, 保留大于分值的 box

  降低阈值的方法为三种:
    1. linear: 1-iou
    2. gaussian:  np.exp( -(iou * iou) / 0.5 )
    3. normal: 1/0
  """
  if len(results) == 0: 
    return []

  keep_list = []
  results = np.array(results)
  tlbrs =  results[:, 0:4]
  scores = results[:,4]
  # 按置信度从大到小进行排序, 返回index
  score_order = scores.argsort()[::-1]
  
  # 相互重叠度很高的box放在一个list里面
  label_list = []
  while len(score_order) > 0:
    max_score_index = score_order[0]
    box = tlbrs[max_score_index]

    keep_inds = []
    same_inds = [ (max_score_index, 1.0) ]

    for ( i, index ) in enumerate( score_order[1:] ):
      iou = get_dice(box, tlbrs[index])
      if( iou <= max_dice ):
        keep_inds.append( i+1 )
      else:
        if method == "linear":
          w = 1 - iou
        elif method == "gaussian":
          w = np.exp( -(iou * iou) / 0.5 )
        else:
          w = 0

        same_inds.append( (index, w) )
    
    score_order = score_order[keep_inds]
    label_list.extend(same_inds)

  new_scores = scores.copy()
  for index,w in label_list:
    new_scores[index] *= w
  
  keep_list = np.where( new_scores >= min_score )[0]
  # pause()
  return tlbrs[keep_list].astype("int"), scores[keep_list]

def preprocess_img(img, height, width):
  img_resize = cv2.resize(img, ( width, height ))  
  img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
  img_data = img_rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255
  img_data= np.ascontiguousarray(img_data)
  return img_data