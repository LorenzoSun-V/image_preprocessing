import cv2
import os
import glob


img_root = "/mnt/shy/农行POC/算法技术方案/demo_1021/加钞间/demo/image2"
img_dirs = []
img_dirs = sorted(glob.glob(os.path.join(img_root, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vw = cv2.VideoWriter('/mnt/shy/农行POC/算法技术方案/demo_1021/加钞间/demo/demo2.mp4', fourcc, 5, (1280, 720),)
for img_dir in img_dirs:
    img = cv2.imread(img_dir)
    vw.write(img)


