import cv2
import os
import glob


img_root = "/mnt/shy/农行POC/abc_data/第九批1116/badcase_1129"
img_dirs = []
#　sorted(glob.glob(os.path.join(img_root, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))
img_dirs = glob.glob(os.path.join(img_root, '*.png'))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter('/mnt/shy/农行POC/abc_data/第九批1116/badcase_1129/demo.mp4', fourcc, 5, (1280, 720),)
for img_dir in img_dirs:
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (1280, 720))

    vw.write(img)


