import os
import glob
import json
import shutil


img_root_path = "/mnt2/shy2/x_ray/x光机图像_64393"
save_root_path = "/mnt2/shy2/x_ray/x光机图像_64393分类/JPEGImages"
save_label = "/mnt2/shy2/x_ray/x光机图像_64393分类/对应图片"
img_folders = os.listdir(img_root_path)
for img_folder in img_folders:
    if img_folder == "__MACOSX":
        continue
    print(img_folder)
    json_file = glob.glob(os.path.join(img_root_path, img_folder, "*.json"))[0]
    img_files = glob.glob(os.path.join(img_root_path, img_folder, "*.jpg"))
    for img_file in img_files:
        if os.path.basename(img_file).startswith("take"):
            img_nolabel = img_file
        else:
            img_label = img_file

    with open(json_file, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        name = json_data["suspicious_cargo_name"]
    if name == '磁性':
        name = 'magnetic'
    elif name == '电池':
        name = 'battery'
    elif name == '粉末':
        name = 'powder'
    elif name == '其他':
        name = 'other'
    elif name == '液体':
        name = 'liquid'
    img_nolabel_name = name + '_' + os.path.basename(img_nolabel)
    img_label_name = name + '_' + os.path.basename(img_label)
    save_nolabel_folder = os.path.join(save_root_path, name)
    save_label_folder = os.path.join(save_label, name)
    if not os.path.exists(save_nolabel_folder):
        os.makedirs(save_nolabel_folder)
    if not os.path.exists(save_label_folder):
        os.makedirs(save_label_folder)
    save_nolabel_path = os.path.join(save_nolabel_folder, img_nolabel_name)
    save_label_path = os.path.join(save_label_folder, img_nolabel_name)
    shutil.copyfile(img_label, save_label_path)
    shutil.copyfile(img_nolabel, save_nolabel_path)
