import os
import shutil
import glob
import torch
import torch.nn.functional as F
import numpy as np


folders = os.listdir("/mnt2/private_data/郑州云踪e/train_data/train/招行_202005视频_工作人员_头肩_工作人员以外人员")
folders = list(filter(lambda name: name.startswith("ZH"), folders))
for folder in folders:
    path = "/mnt2/private_data/郑州云踪e/train_data/train/招行_202005视频_工作人员_头肩_工作人员以外人员/{}".format(folder)
    save_path = "/mnt2/private_data/郑州云踪e/test_demo"
    img_dirs = glob.glob(os.path.join(path, "*.jpg"))
    length = len(img_dirs) // 20
    img_dirs = list(filter(lambda name: img_dirs.index(name) % length == 0, img_dirs))
    for i, img_dir in enumerate(img_dirs):
        # if i % 3 == 0:
        img_name = os.path.basename(img_dir)
        dst_path = os.path.join(save_path, img_name)
        shutil.move(img_dir, dst_path)

# x = np.array([0.7452, -3.3458, -1.7098, 4.79005, -0.94021])
# x_tensor = torch.from_numpy(x)
# out = F.softmax(x_tensor, dim=0)
# print(out)
