import cv2
import numpy as np


label_list = ["knife", "scissors", "sharpTools", "expandableBaton", "smallGlassBottle", "electricBaton",
              "plasticBeverageBottle", "plasticBottleWithaNozzle", "electronicEquipment", "battery",
              "seal", "umbrella"]
color_list = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
              (240, 32, 160), (79, 79, 47), (147, 20, 255), (179, 222, 245), (86, 114, 255), (197, 181, 255)]
img = np.ones_like((1200, 1200, 3)) * 255
# for i, label in enumerate(label_list):
#     color = color_list[ label_list.index( label ) ]
#     if i < 6:
#         cv2.putText(img, label, (0, 100*i), 0, 0.5, list(color), thickness=1, lineType=cv2.LINE_AA)
#     else:
#         cv2.putText(img, label, (600, 100 * i), 0, 0.5, list(color), thickness=1, lineType=cv2.LINE_AA)
cv2.namedWindow("11")
cv2.imshow("11", img)
cv2.waitKey(0)
cv2.destroyAllWindows()