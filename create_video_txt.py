import os
import glob


def create_txt_withFolder(video_path):
    # video_path = "/mnt1/shy1/test_jinzhai/raw_video/hedao"
    video_folders = os.listdir(video_path)
    for video_folder in video_folders:
        current_video_folder = os.path.join(video_path, video_folder)
        videos = os.listdir(video_path)
        # video_dirs = glob.glob(os.path.join(current_video_folder, "*.mp4"))
        for video in videos:
            current_video = os.path.join(current_video_folder, video)
            video_suffix = os.path.basename(current_video).split(".")[-1]
            txt_name = os.path.basename(current_video).replace(video_suffix, "txt")
            txt_path = os.path.join(current_video_folder, txt_name)
            with open(txt_path, "wb") as f:
                pass


def create_txt_withoutFolder(video_path):
    # video_path = "/mnt1/shy1/test_jinzhai/raw_video/hedao"
    videos = os.listdir(video_path)
    for video in videos:
        current_video = os.path.join(video_path, video)
        video_suffix = os.path.basename(current_video).split(".")[-1]
        if video_suffix in ["mp4", "MP4"]:
            txt_name = os.path.basename(current_video).replace(video_suffix, "txt")
            txt_path = os.path.join(video_path, txt_name)
            with open(txt_path, "wb") as f:
                pass


if __name__ == "__main__":
    video_path = "/mnt/shy/农行POC/abc_data/第六批1018/full"
    create_txt_withoutFolder(video_path)
    # create_txt_withFolder()

