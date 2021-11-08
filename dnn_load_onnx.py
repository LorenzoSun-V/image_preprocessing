import os

import cv2
import numpy as np
import glob


def soft_resize( src, h, w ):
    tmp = cv2.resize( src, (2*w, 2*h) ).astype( np.float32 )
    A = tmp[0:2*h:2, 0:2*w:2, :]
    B = tmp[0:2*h:2, 1:2*w:2, :]
    C = tmp[1:2*h:2, 0:2*w:2, :]
    D = tmp[1:2*h:2, 1:2*w:2, :]
    return ( 0.25 / 255 ) * ( A + B + C + D )


def sigmoid( src ):
    return 1.0 / ( 1.0 + np.exp( -src ) )


def infer_img(img_dir, net):
    if type(img_dir) == str:
        #  如果传入的是图片路径，那么就按照图片方式处理
        srcimg = cv2.imread(img_dir)
    else:
        #  如果传入的是图片np数组，那么就按照video方式处理
        srcimg = img_dir
    # 前处理
    resize_img = cv2.resize(srcimg, (1024, 576))
    img = resize_img.astype(np.float32) / 255.0
    # img = soft_resize( srcimg, 576, 1024 )

    # 模型推理
    blob = cv2.dnn.blobFromImage(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    outs = outs[0][0][0]
    # 后处理
    # outs[outs<0] = 0
    # outs[outs>0] = 255
    # result = outs[0][0]
    outs = cv2.resize(outs, (1024, 576))
    _, binary = cv2.threshold(outs, 0, 255, cv2.THRESH_BINARY)
    binary = np.array(binary, np.uint8)
    # 由于这里的img范围是(0, 1), 但img的mask范围是(0, 255), 所以统一范围避免之后从float32转到uint8时精度损失造成图片颜色不对
    # img_with_mask dtype=np.uint8
    img_with_mask = addmask2img(resize_img, binary)
    return img_with_mask


def load_onnx(is_video):
    onnx_path = "/mnt1/algorithms/yanghao/seg_models/河道/hedao_seg_model_W0.onnx"
    try:
        net = cv2.dnn.readNet(onnx_path)
        print('read sucess')
    except:
        print('read failed')
    if not is_video:
        img_root_path = "/mnt/shy/sjh/YOLOP-main/hedao_img_test"
        img_save_path = "/mnt/shy/sjh/YOLOP-main/hedao_img_test/result"
        img_dirs = glob.glob(os.path.join(img_root_path, "*.jpg"))
        for img_dir in img_dirs:
            img_result = infer_img(img_dir, net)
            img_name = os.path.basename(img_dir)
            current_img_save_path = os.path.join(img_save_path, img_name)
            cv2.imwrite(current_img_save_path, img_result)
            print("{} is done".format(img_name))
    else:
        video_root_path = "/mnt1/shy1/test_jinzhai/hedao_cut"
        video_demo_save_path = "/mnt1/shy1/test_jinzhai/hedao_cut_demo"
        video_dirs = glob.glob(os.path.join(video_root_path, "*.mp4"))
        for i, video_dir in enumerate(video_dirs):
            print("{} / {}".format(i, len(video_dirs)))
            video_name = os.path.basename(video_dir)
            video_capture = cv2.VideoCapture(video_dir)
            save_video_path = "{}/{}".format(video_demo_save_path, video_name)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(save_video_path, fourcc, fps=25,
                                           frameSize=(1024, 576))
            frame_nums = video_capture.get(7)

            # 帧数记录
            st = 0
            # 确定跳帧数
            if frame_nums <= 800:
                skip_frame = 5
            elif (frame_nums > 800) and (frame_nums <= 1600):
                skip_frame = 15
            else:
                skip_frame = 25
            skip_frame = 3
            while True:
                ret, im0 = video_capture.read()
                st += 1
                if ret:
                    if st % skip_frame == 0:
                        img_result = infer_img(im0, net)
                        # img_result = cv2.resize(img_result, (1920, 1080))
                        video_writer.write(img_result)
                        # print("write to {}".format(save_video_path))

                        # 存图片
                        # save_video_path = "{}/{}_{}".format(video_demo_save_path, st, video_name.replace("mp4", "jpg"))
                        # cv2.imwrite(save_video_path, img_result)
                else:
                    break


def addmask2img(img, mask):
    color_area = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_area[mask==255] = [255, 0, 0]
    color_seg = color_area
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    # img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    # mask = np.expand_dims(mask, axis=2)
    # img2 = mask + img
    return img


if __name__ == "__main__":
    load_onnx(is_video=True)