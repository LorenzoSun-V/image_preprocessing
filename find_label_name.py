'''
Author: shy
Description: 图片标注质量检查
LastEditTime: 2021-08-12 18:12:14
'''
import os, cv2, random
from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def load_voc_xml( path ):
    xml = open( path, "r" )
    tree = ET.parse( xml )
    xml.close()
    return tree

def find_files(folder, suffix):
    files = []
    files.extend( glob(str(folder / "**.{}".format(suffix))) )
    if len(files) == 0:
        files = find_folder_files(folder, suffix)

    print("{} find {} {} files".format( folder, len(files), suffix ) )
    return files

def find_folder_files(folder, suffix):
    sub_dirs = [x for x in folder.iterdir() if x.is_dir()]
    loop = tqdm(sub_dirs, total=len(sub_dirs))
    files = []
    for sub_dir in loop:
        files.extend( glob(str(sub_dir / "**.{}".format(suffix))) )
    print("{} find {} {} files".format( folder, len(files), suffix ) )
    return files

def check_files(src_names, dst_names):
    lost_names = np.setdiff1d(src_names, dst_names)
    if len(lost_names) > 0:
        print(" {} 文件不一致： \n {}".format( len(lost_names), lost_names ) )
    else:
        print("文件一致")
    return lost_names

# 检查 对应的图片和标注文件  是否存在缺失情况
def find_redundant_files(xml_files, img_files):
    xml_names = np.array( [ x.split("Annotations")[1][:-4] for x in xml_files ] )
    img_names = np.array( [ x.split("JPEGImages" )[1][:-4] for x in img_files ] )
    common_names = np.intersect1d(xml_names, img_names) # 两者共同的图片
    print(" 符合条件的 图片和标注文件有 {}".format( len(common_names) ) )

    _ = check_files(xml_names, img_names) # xml文件 找不到对应的图片
    _ = check_files(img_names, xml_names) # 图片 找不到对应的xml文件

    random.shuffle(common_names)

    xml_dir = xml_files[0].split("Annotations")[0]
    img_dir = img_files[0].split("JPEGImages" )[0]
    img_paths, xml_paths = [], []
    for common_name in common_names:
        xml_paths.append( xml_dir + "Annotations/" + common_name + ".xml" )
        img_paths.append( img_dir + "JPEGImages/"  + common_name + ".jpg" )

    return img_paths, xml_paths


# 检查图片标注情况
def inspect_img(img_paths, xml_paths, label_list, color_list):

    for img_path, xml_path in zip(img_paths, xml_paths):

        img = cv2.imread( img_path )
        tree = load_voc_xml( xml_path )
        rt = tree.getroot()

        for obj in rt.findall( "object" ):
            name = obj.find( "name" ).text

            if name not in label_list:
                os.remove(xml_path)
                os.remove(img_path)

            # if name in label_list:
                # bbox = obj.find( "bndbox" )
                # xmin = int(float(bbox.find( "xmin" ).text))
                # ymin = int(float(bbox.find( "ymin" ).text))
                # xmax = int(float(bbox.find( "xmax" ).text))
                # ymax = int(float(bbox.find( "ymax" ).text))
                # color = color_list[ label_list.index( name ) ]
                # cv2.rectangle( img, (xmin, ymin), (xmax, ymax), color, 2 )
                # lms = obj.find( "lm" )
                # if lms is not None:
                #     x1 = int(float(lms.find( "x1" ).text))
                #     y1 = int(float(lms.find( "y1" ).text))
                #     x2 = int(float(lms.find( "x2" ).text))
                #     y2 = int(float(lms.find( "y2" ).text))
                #     x3 = int(float(lms.find( "x3" ).text))
                #     y3 = int(float(lms.find( "y3" ).text))
                #     x4 = int(float(lms.find( "x4" ).text))
                #     y4 = int(float(lms.find( "y4" ).text))
                #     x5 = int(float(lms.find( "x5" ).text))
                #     y5 = int(float(lms.find( "y5" ).text))
                #     points_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5),]
                #     for point in points_list:
                #         cv2.circle(img, point, 1, color, 2)

        # cv2.namedWindow( 'det_result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('det_result', 1080, 720)
        # # cv2.resizeWindow('det_result', 640, 480)
        # cv2.imshow("det_result", img)
        # print( xml_path )
        #
        # ch = cv2.waitKey(0) # 按enter键切换下一张图片
        # # 按 ese或q键退出显示
        #
        # if ch == "d":
        #     cv2.destroyAllWindows()  # release windows
        #     os.remove(xml_path)
        #     os.remove(img_path)
        #     print("delete {img_path}")
        # elif ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     break

def find_labels(img_paths, xml_paths, label_list, new_dir):

    new_xml_dir = Path( new_dir + "Annotations/" )
    new_img_dir = Path( new_dir + "JPEGImages/" )

    for img_path, xml_path in zip(img_paths, xml_paths):
        tree = load_voc_xml( xml_path )
        rt = tree.getroot()

        for obj in rt.findall( "object" ):
            name = obj.find( "name" ).text
            if name in label_list:
                new_img_path = new_img_dir / Path(img_path).name
                new_xml_path = new_xml_dir / Path(xml_path).name
                shutil.copy(img_path, new_img_path)
                shutil.copy(xml_path, new_xml_path)


def check_img(img_paths):
    loop = tqdm(img_paths, total= len(img_paths))

    for img_path in loop:
        im = cv2.imread(img_path)
        if im is None: print("wrong f{img_path}, can't read")

if __name__ == "__main__":
    # root = "/mnt1/0_各项目标定结果/目标检测/头_脸_五官/face_only/人脸数据_算法研究用/"
    # root = "/mnt1/0_各项目标定结果/目标检测/头_脸_五官_头肩/face_only/人脸检测评价数据_招行中威修正后数据/"
    # root = "/mnt2/general_AI_datasets/airplane/airplane_project/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/"
    # root = "/mnt1/shy1/label_task/vehicle_0419/"
    # root = "/mnt1/shy1/label_task/vehicle_project/"
    # root = "/mnt1/0_各项目标定结果/目标检测/机场/1类别/算法研究用/20210412/"
    # root = "/mnt1/0_各项目标定结果/目标检测/机场/1类别/算法研究用/20210419/"

    ## 通用模型 车辆
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_clean/private/10分类_22775/" # 22775
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_clean/private/14分类_41146/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_clean/private/东海大桥_1946/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_clean/private/2分类汽车/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/14分类/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/2分类_汽车/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/5分类_吊车/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/5分类_钻头/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/东海大桥/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/火车等/"
    # root = "/mnt2/general_AI_datasets/vehicle/BDD100K/" # 无需清洗 79292
    # root = "/mnt1/0_各项目标定结果/目标检测/车/算法研究用/" # 无需清洗 79292
    # root = "/mnt2/general_AI_datasets/vehicle/UrbanTraffic/" # 标注质量差 78946
    # root = "/mnt2/general_AI_datasets/vehicle/kitti_misc/" # 无需清洗 748
    # root = "/mnt2/general_AI_datasets/vehicle/kitti_no_misc/" # 无需清洗 6050
    # label_list = [ "car", "truck", "bus",  ]

    # label_list = [ "car", "truck", "bus", "Misc", ]



    # 通用模型 飞机
    # root = "/mnt1/shy1/airplane_project/"
    # root = "/mnt1/0_各项目标定结果/目标检测/机场/1类别/算法研究用/"
    # root = "/mnt1/0_各项目标定结果/目标检测/机场/1类别/广州白云机场/PA白云机场_AI/"
    # root = "/mnt2/shy2/face_dataset/detect/public/widerface_voc_train/"
    # root = "/mnt/shy//widerface_voc_train/"
    # root = "/mnt2/shy2/face_dataset/detect/public/widerface_voc_val/"
    # root = "/mnt1/shy1/fire_detect/"
    # root = "/mnt1/0_各项目标定结果/目标检测/车/算法研究用/"
    # root = "/mnt1/shy1/COCO_airplane_clean/"
    # root = "/mnt2/general_AI_datasets/airplane/COCO_airplane/"
    # root = "/mnt2/general_AI_datasets/vehicle/COCO_vehicle/"
    # root = "/mnt2/general_AI_datasets/vehicle/vehicle_project/"
    # label_list = [ "face", ]
    # label_list = [ "airplane", ]

    # root = "/mnt1/0_各项目标定结果/目标检测/机场/1类别/广州白云机场/GZHG_20200107/"
    # label_list = [ "airplane", ]


    # root = "/mnt2/general_AI_datasets/airport/bridge/21类_bridge/"
    # root = "/mnt2/general_AI_datasets/airport/bridge/bridge_0518/"
    # label_list = [ "bridge" ,  "working_bridge", ]



    # 火焰检测
    # root = "/mnt2/general_AI_datasets/objects/fire_smoke/fire_detection_VOC2020/"
    # root = "/mnt1/shy1/fire_detect/"
    # root = "/mnt/shy/fire/fire_seg_2000/"
    # label_list = [ "Fire", "fire", "yanwu", "smoke", "smoking", ]
    # root = "/mnt2/general_AI_datasets/fire/fire_det_3472/"
    # label_list = [ "fire",  ]

    # 手势检测
    # root = "/mnt2/general_AI_datasets/hand/datasets_TVCOCO_hand_train/"
    # label_list = [ "hand",  ]

    # 店员检测
    # root = "/mnt1/shy1/客流数人_头肩_人体_标牌/20210415/"
    # root = "/mnt1/shy1/客流数人_头肩_人体_标牌/20210421/"
    # label_list = [ "person", "head_shoulder", "badge" ]

    # 电瓶车检测
    # root = "/mnt2/general_AI_datasets/motorcycle/10分类/"
    # root = "/mnt2/general_AI_datasets/motorcycle/14分类/"
    # # root = "/mnt2/general_AI_datasets/motorcycle/COCO_det/"
    # label_list = [ "motorcycle", ]

    ## objects365
    # root = "/mnt2/shy2/detect_dataset/2021/objects365_2020_person/"
    # label_list = [ "Person", ]

    # root = "/mnt2/shy2/detect_dataset/2021/objects365_2020_airplane/"
    # label_list = [ "Airplane", ]

    # root = "/mnt2/shy2/detect_dataset/2021/car_truck_bus/"
    # label_list = [ "Car", "Bus", "Truck", "Fire Truck", "Heavy Truck", "Pickup Truck", ]

    # root = "/mnt2/shy2/detect_dataset/2021/objects365_2020_motorcycle/"
    # label_list = [ "Motorcycle",  ]

    # root = "/mnt2/shy2/detect_dataset/2021/objects365_2020_fire_extinguisher/"
    # label_list = [ "Fire Extinguisher", ]


    # root = "/mnt2/shy2/detect_dataset/2021/objects365_2020_cigar/"
    # label_list = [ "Cigar/Cigarette", ]


    # 灭火器检测
    # root  = "/mnt2/shy2/detect_dataset/2021/Objects365/"
    # root  = "/mnt2/shy2/detect_dataset/2021/fire_extinguisher/"
    # label_list = [ "fire extinguisher", ]

    # root =  "/mnt1/shy1/警务督察误检/测试审讯/output_test/"
    # root = "/mnt1/shy1/兰州电网误检/result/三角城误检13点42/"
    # root = "/mnt1/shy1/兰州电网误检/result/庄浪路13点27/"
    # root = "/mnt1/shy1/农行POC视频_210526/需要标定文件/demo_xml/"

    # root = "/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/农业银行第三批_轮椅拐杖/"
    # root = "/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/农业银行第三批_拐杖轮椅7.23/"
    # root = "/mnt1/shy1/农行bad_case/拐杖轮椅已修正8.3/"
    root = "/mnt1/0_各项目标定结果/目标检测/人_含轮椅/招行/"
    # root = "/mnt1/0_各项目标定结果/目标检测/公司职员/农业银行/农业银行第三批_轮椅拐杖/"

    xml_dir = Path( root + "Annotations/" )
    img_dir = Path( root + "JPEGImages/" )

    # 需要检查的类别
    # label_list         = [ "wheelchair", "crutch_Y", "crutch_1", ]
    # label_list = [ "person", "head_shoulder", "face", "hand", ]
    # label_list = [ "airplane", ]
    # label_list = [ "face", ]
    # label_list = ["person", "other"]
    label_list = ["crutch"]
    color_list = [ (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255),] #cv2 bgr

    # 获取 针对二级文件夹的图片路径
    xml_files = find_files(xml_dir, "xml")
    img_files = find_files(img_dir, "jpg")
    # 检查 对应的图片和标注文件  是否存在缺失情况
    img_paths, xml_paths = find_redundant_files(xml_files, img_files)

    # 检查图片标注情况
    inspect_img(img_paths, xml_paths, label_list, color_list)
    # find_labels(img_paths, xml_paths, label_list, new_dir="/mnt2/shy2/detect_dataset/2021/fire_extinguisher/")