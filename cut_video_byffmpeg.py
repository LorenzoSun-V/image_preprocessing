import os
import glob


video_path = "/mnt/shy/农行POC/第三批0716/part1_cut_video/C57_3_0617_1230-1330"
save_path = "/mnt/shy/农行POC/第三批0716/part1_cut_video/C57_cleaner"
video_dirs = glob.glob(os.path.join(video_path, "*.mp4"))

time_lists = {
    "C57_3_0617_1230-1330_003732--003748.mp4": ["00:00:00-00:00:14"],
    "C57_3_0617_1230-1330_003818--003836.mp4": ["00:00:00-00:00:15"],
    "C57_3_0617_1230-1330_004330--004350.mp4": ["00:00:00-00:00:15"],
    "C57_3_0617_1230-1330_004746--004810.mp4": ["00:00:00-00:00:20"],
    "C57_3_0617_1230-1330_004826--005030.mp4": ["00:00:20-00:00:32", "00:01:14-00:01:25"]
}

for video_dir in video_dirs:
    video_name = os.path.basename(video_dir)
    save_root_name = video_name.split(".")[0]
    if video_name in time_lists:
        time_list = time_lists[video_name]
        for i, each_time in enumerate(time_list):
            begin_time = each_time.split("-")[0]
            end_time = each_time.split("-")[-1]
            save_name = "{}_{}.mp4".format(save_root_name, i)
            save_dst = os.path.join(save_path, save_name)
            # os.system("ffmpeg  -i {} -vcodec copy -acodec copy -ss {} -to {} {} -y".format(
            #     video_dir, begin_time, end_time, save_dst
            # ))
            os.system("ffmpeg -ss {} -to {} -i {} -vcodec copy -acodec copy {}".format(
                begin_time, end_time, video_dir, save_dst))
            print("ffmpeg  -i {} -vcodec copy -acodec copy -ss {} -to {} {} -y".format(
                video_dir, begin_time, end_time, save_dst))




