import cv2
import numpy as np

"""
    鼠标点击，获取当前坐标值
"""

img = cv2.imread("/mnt/shy/农行POC/算法技术方案/demo_1021/窗口工作状态/1.jpg")


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break

cv2.waitKey(0)
cv2.destroyAllWindow()
