

def time2time(start):
    all = int(start[0:2]) * 3600 + int(start[2:4]) * 60 + int(start[4:6])
    # hour = all // 3600
    # minute = (all % 3600) // 60
    # second = (all % 3600) % 60
    # start = str(hour).zfill(2) + str(minute).zfill(2) + str(second).zfill(2)
    return all


path = "/mnt/shy/农行POC/第一批0610/raw_video/uncut"
import os
import glob
dirs = glob.glob(os.path.join(path, "*.mp4"))
for dir in dirs:
    video_name = os.path.basename(dir)
    txt_name = dir.split(".")[0] + ".txt"
    txt_path = os.path.join(os.path.dirname(dir), txt_name)
    with open(txt_path, "w") as f:
        f.write("")

# source_txt = "/mnt/shy/农行POC/第三批0722/C65_5_0702_1030-1230.txt"
# start_times = []
# stop_times = []
# diff = 0
# with open(source_txt, 'r') as f:
#     for line in f.readlines():
#         line = line.strip('\n')
#         print(line)
#         start_time, stop_time = line.split(' ')[:]
#         start_times.append(start_time)
#         stop_times.append(stop_time)
# for i in range(len(start_times)):
#     start = start_times[i]
#     stop = stop_times[i]
#     print(start, stop)
#     start_time = time2time(start)
#     stop_time = time2time(stop)
#     diff += (stop_time - start_time)
# print("diff:{}".format(diff))
# a = "C26、C50、C56、C57、C58、C92、C08、C09、C39、C95、C96、C45、C46、C48、C61、C65、C119"
# b = a.split("、")
# print(b)
# b = sorted(b, key=lambda x:x[1:])
# c = ""
# for i in b:
#     c += "、{}".format(i)
# print(c)