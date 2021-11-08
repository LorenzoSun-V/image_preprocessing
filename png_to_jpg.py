import os
import cv2


png_list = ["/mnt1/shy1/电瓶车检测标定/images/22.481ca699ed4e4307a9a75977484118da.png",
            "/mnt1/shy1/电瓶车检测标定/images/59.20180111042946133.png",
            "/mnt1/shy1/电瓶车检测标定/images/176.1600061720841941.png",
            "/mnt1/shy1/电瓶车检测标定/images/213.f714194e54aa670_w650_h867.png"]

for file in png_list:
    if file.endswith(".png"):
        img = cv2.imread( file )
        dst = file[0:len(file)-len(".png")] + ".jpg"
        cv2.imwrite( dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        # os.remove( path )
# for root, dirs, files in os.walk("JPEGImages"):
#     for f in files:
#         path = os.path.join(root, f)
#
#         if path.endswith( ".png" ):
#             img = cv2.imread( path )
#             dst = path[0:len(path)-len(".png")] + ".jpg"
#             cv2.imwrite( dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
#             os.remove( path )
#
#         if path.endswith( ".jpeg" ) or path.endswith( ".JPEG" ) or path.endswith( ".JPG" ):
#             dst = os.path.splitext(path)[0] + ".jpg"
#             os.rename( path, dst )


