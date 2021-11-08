import os
import glob
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
# from pydub import AudioSegment


def time2time(start, temp_time):
    all = int(start[0:2]) * 3600 + int(start[2:4]) * 60 + int(start[4:6]) - temp_time
    # hour = all // 3600
    # minute = (all % 3600) // 60
    # second = (all % 3600) % 60
    # start = str(hour).zfill(2) + str(minute).zfill(2) + str(second).zfill(2)
    return all


def clip_video(source_file, target_file, start_time, stop_time, video_fps):
    """
    利用moviepy进行视频剪切
    :param source_file: 原视频的路径，mp4格式
    :param target_file: 生成的目标视频路径，mp4格式
    :param start_time: 剪切的起始时间点（第start_time秒）
    :param stop_time: 剪切的结束时间点（第stop_time秒）
    :return:
    """
    validate_file(source_file)
    source_video = VideoFileClip(source_file, fps_source="fps")
    video = source_video.subclip(int(start_time), int(stop_time))  # 执行剪切操作
    video.write_videofile(target_file, fps=video_fps)  # 输出文件


def clip_audio(source_file, target_file, start_time, stop_time):
    """
    利用pydub进行音频剪切。pydub支持源文件为 mp4格式，因此这里的输入可以与视频剪切源文件一致
    :param source_file: 原视频的路径，mp4格式
    :param target_file: 生成的目标视频路径，mp4格式
    :param start_time: 剪切的起始时间点（第start_time秒）
    :param stop_time: 剪切的结束时间点（第stop_time秒）
    :return:
    """
    validate_file(source_file)
    audio = AudioSegment.from_file(source_file, "mp4")
    audio = audio[start_time * 1000: stop_time * 1000]
    audio_format = target_file[target_file.rindex(".") + 1:]
    audio.export(target_file, format=audio_format)


def combine_video_audio(video_file, audio_file, target_file, delete_tmp=False):
    """
    利用 ffmpeg将视频和音频进行合成
    :param video_file:
    :param audio_file:
    :param target_file:
    :param delete_tmp: 是否删除剪切过程生成的原视频/音频文件
    :return:
    """
    validate_file(video_file)
    validate_file(audio_file)
    # 注：需要先指定音频再指定视频，否则可能出现无声音的情况
    command = "ffmpeg -y -i {0} -i {1} -vcodec copy -acodec copy {2}".format(audio_file, video_file, target_file)
    os.system(command)
    if delete_tmp:
        os.remove(video_file)
        os.remove(audio_file)


def clip_handle(source_file, target_file, start_time, stop_time, video_fps, tmp_path=None, delete_tmp=False):
    """
    将一个视频文件按指定时间区间进行剪切
    :param source_file: 原视频文件
    :param target_file: 目标视频文件
    :param start_time: 剪切的起始时间点（第start_time秒）
    :param stop_time: 剪切的结束时间点（第stop_time秒）
    :param tmp_path: 剪切过程的文件存放位置
    :param delete_tmp: 是否删除剪切生成的文件
    :return:
    """
    # 设置临时文件名
    if tmp_path is None or not os.path.exists(tmp_path):
        # 如果没有指定临时文件路径，则默认与目标文件的位置相同
        tmp_path = target_file[: target_file.rindex("/") + 1]
    target_file_name = target_file[target_file.rindex("/") + 1: target_file.rindex(".")]
    tmp_video = tmp_path + target_file_name + ".mp4"
    # tmp_video = tmp_path + "v_" + target_file_name + ".mp4"
    # tmp_audio = tmp_path + "a_" + target_file_name + ".mp4"

    # 执行文件剪切及合成
    clip_video(source_file, tmp_video, start_time, stop_time, video_fps)
    # clip_audio(source_file, tmp_audio, start_time, stop_time)
    # combine_video_audio(tmp_video, tmp_audio, target_file, delete_tmp)


def validate_file(source_file):
    if not os.path.exists(source_file):
        raise FileNotFoundError("没有找到该文件：" + source_file)


def example():
    """
    测试例子
    :return:
    """
    video_path = "/mnt/shy/农行POC/第三批0716"
    save_path = "/home/user/data/C58_badcase"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    video_dirs = glob.glob(os.path.join(video_path, "C58_4_0617_1230-1330.mp4"))
    from timeline import C58_badcase
    time_lists = C58_badcase
    # "C92_2_0702_1030-1230.mp4", "C56_2_0702_1030-1230.mp4", "C57_3_0617_1230-1330.mp4",
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        save_root_name = video_name.split(".")[0]
        if video_name in time_lists:
            time_list = time_lists[video_name]
            for i, each_time in enumerate(time_list):
                save_name = "{}_{}.mp4".format(save_root_name, i)
                save_dst = os.path.join(save_path, save_name)
                begin_time = each_time.split("-")[0].split(":")
                end_time = each_time.split("-")[-1].split(":")
                start = "".join((x for x in begin_time))
                stop = "".join((y for y in end_time))
                print(start, stop)
                start_time = int(start[0:2]) * 3600 + int(start[2:4]) * 60 + int(start[4:6])
                stop_time = int(stop[0:2]) * 3600 + int(stop[2:4]) * 60 + int(stop[4:6])
                    # print(start_time, stop_time)
                # 设置目标文件名
                # target_name = video_name[:-4] + "_" + str(start) + "--" + str(stop)
                # target_file = os.path.join(root_path, 'cut', target_name + ".mp4")
                # 处理主函数
                clip_handle(video_dir, save_dst, start_time, stop_time)


def example2(root_path, video_dirs, temp_time):
    """
    测试例子
    :return:
    """

    save_file = "/mnt/shy/农行POC/算法技术方案/demo_1021/窗口工作状态"
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    for i, video_dir in enumerate(video_dirs):
        capture = cv2.VideoCapture(video_dir)
        video_fps = capture.get(5)
        print("----------------------------------------------------")
        print("--------{} / {}---------".format(i, len(video_dirs)-1))
        print("-----------{}-----------".format(video_dir))
        print("----------------------------------------------------")
        # try:
        #     temp_time = temp_time_all[os.path.basename(video_dir)]
        # except:
        #     continue
        video_name = os.path.basename(video_dir)[:-4]
        # video_folder_name = video_name.split("_")[0]
        # save_sub_file = os.path.join(save_file, video_folder_name)
        # if not os.path.exists(save_sub_file):
        #     os.makedirs(save_sub_file)
        txt_name = "".join((video_name, '.txt'))
        source_txt = os.path.join(os.path.dirname(video_dir), txt_name)
        if os.path.exists(source_txt):
            start_times = []
            stop_times = []
            with open(source_txt, 'r') as f:
                for line in f.readlines():
                    if line:
                        line = line.strip('\n')
                        print(line)
                        start_time, stop_time = line.split(' ')[0], line.split(' ')[-1]
                        start_times.append(start_time)
                        stop_times.append(stop_time)
            for i in range(len(start_times)):
                start = start_times[i]
                stop = stop_times[i]
                print(start, stop)
                start_time = time2time(start, temp_time)
                stop_time = time2time(stop, temp_time)
                # stop_time = (stop_time - start_time) / 2 + start_time
                # 设置目标文件名
                target_name = video_name + "_" + str(start) + "--" + str(stop)
                target_file = os.path.join(save_file, target_name + ".mp4")
                # 处理主函数
                if not os.path.exists(target_file):
                    clip_handle(video_dir, target_file, start_time, stop_time, video_fps)

    return save_file


if __name__ == "__main__":
    root_path = '/mnt/shy/农行POC/abc_data/第六批1020/cut/C26'

    # video_dirs = []
    # folder_names = os.listdir(root_path)
    # for folder_name in folder_names:
    #     current_folder_path = os.path.join(root_path, folder_name)
    #     video_dirs.extend(glob.glob(os.path.join(current_folder_path, "*.mp4")))

    video_dirs = glob.glob(os.path.join(root_path, "*.mp4"))

    # a = [os.path.basename(x) for x in video_dirs]
    # print(a)
    # merge_path = root_path + '/' + str(video_names[0][:3]) + '_badcase'
    temp_time_all = {"C08_2_0929_1010_1020.mp4": 36599,
                     "C08_2_0929_1615_1646.mp4": 58498,
                     "C09_3_0928_1630_1646.mp4": 59399,
                     "C45_3_0928_1545_1635.mp4": 56735,
                     "C45_3_0929_1028_1103.mp4": 37718,
                     "C45_3_0929_1328_1403.mp4": 48522,
                     "C57_3_0928_1350_1446.mp4": 49798,
                     "C128_2_0928_1400_1440.mp4": 50400,
                     "C128_2_0928_1605_1625.mp4": 57900}
    temp_time = 0  # 10:29:58
    # "C92_2_0702_1030-1230.mp4", "C56_2_0702_1030-1230.mp4", "C57_3_0617_1230-1330.mp4",
    video_path = example2(root_path, video_dirs, temp_time)
    # merge_video_ffmepeg(video_path, merge_path)

