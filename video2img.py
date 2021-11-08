import os
import cv2
import time


def v2i(src, out_src, jpg_quality, skip_frame):
    vc = cv2.VideoCapture( src )
    num = 1
    if vc.isOpened():
        for i in range(0, 1000000):
            rval, frame = vc.read()
            if rval is True:
                if (i+1) % skip_frame == 0:
                    path = out_src + "_" + str(num).zfill(4) + ".jpg"
                    num += 1
                    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            else:
                break

    vc.release()


if __name__ == '__main__':

    in_path = "/mnt/shy/农行POC/abc_data/第六批1018/wheelchair"
    out_path = "/mnt/shy/农行POC/abc_data/第六批1018/wheelchair_img"
    jpg_quality = 80
    skip_frame = 3

    for name in os.listdir(in_path):
        # if name == "0823.mp4":
        src = in_path + '/' + name
        out_src = os.path.join(out_path, str(name[:-4]))         # :-4
        # if not os.path.exists(out_src):
        #     os.makedirs(out_src)
        print(out_src)
        v2i(src, out_src, jpg_quality, skip_frame)
