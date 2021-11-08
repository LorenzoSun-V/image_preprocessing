# 用于图像透视变换，并保存

import cv2
import os
import numpy as np
from glob import glob


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def crop_img(img_root_path, img_save_path):
    img_dirs = glob(os.path.join(img_root_path, "*.jpg"))
    for img_dir in img_dirs:
        img = cv2.imread(img_dir)
        H_rows, W_cols = img.shape[:2]
        pts1 = order_points(np.float32([[6808,66], [7607,165], [7613,343], [6811,256]]))
        (tl, tr, br, bl) = pts1
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts1, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        # return the warped image
        # return warped

        img_name = os.path.basename(img_dir)
        warped_save_path = os.path.join(img_save_path, img_name)
        cv2.imwrite(warped_save_path, warped)
        # img2 = img[66: 343, 6808:7613]

        # watch the img
        # cv2.namedWindow("fff", cv2.WINDOW_NORMAL)
        # cv2.imshow("fff", warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    img_root_path = "/mnt1/shy1/华东空管_全景拼接/时间图片/0823"
    img_save_path = "/mnt1/shy1/华东空管_全景拼接/时间图片/时间小图"
    crop_img(img_root_path, img_save_path)
