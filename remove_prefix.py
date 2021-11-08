import glob
import os


img_root = "/home/user/PycharmProjects/classification/mobilenet/dataset/badcase_test/error(明显分类错误)"
img_lists = ["bank_staff", "cleaner", "money_staff", "person", "security_staff"]
for img_list in img_lists:
    img_path = os.path.join(img_root, img_list)
    img_dirs = glob.glob(os.path.join(img_path, "*.jpg"))
    for img_dir in img_dirs:
        img_name = os.path.basename(img_dir)
        if img_name[:10]=="bank_staff":
            new_img_name = img_name[10:]
        elif img_name[:7]=="cleaner":
            new_img_name = img_name[7:]
        elif img_name[:11]=="money_staff":
            new_img_name = img_name[11:]
        elif img_name[:6]=="person":
            new_img_name = img_name[6:]
        elif img_name[:14]=="security_staff":
            new_img_name = img_name[14:]
        new_img_path = os.path.join(os.path.dirname(img_dir), new_img_name)
        os.rename(img_dir, new_img_path)
