import os
import glob


video_path = "/mnt1/shy1/test_shequ/fall_demo/badcase_cut_0913"
video_dirs = glob.glob(os.path.join(video_path, "*.mp4"))
with open("file.txt", "wb") as f:
    for video_dir in video_dirs:
        content = "file  {}\n".format(video_dir).encode()
        f.write(content)

os.system("ffmpeg -f concat -safe 0 -i file.txt -c copy /mnt1/shy1/test_shequ/fall_demo/badcase_merge_0913/output.mp4")