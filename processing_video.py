import os, sys, shutil
import cv2
import numpy as np
from multiprocessing import *
from numpy import random

current_path = os.getcwd()


# task 1: split videos in the path into dirs with frames
# task 2: make .npy ground truth according to settings

if task == 1:

    def get_frame_from_video(path):
        """Convert all videos to frames"""
        videos = os.listdir(path)
        for video_name in videos:
            file_name = video_name.split('.')[0]
            folder_name = os.path.join(path, file_name)
            os.makedirs(folder_name, exist_ok=True)
            print("Split {} to {}".format(os.path.join(path, video_name), folder_name))

            vc = cv2.VideoCapture(os.path.join(path, video_name))  # 读入视频文件
            c = -1
            rval = vc.isOpened()

            while rval:  # read frames recurrently
                c = c + 1
                rval, frame = vc.read()
                pic_path = folder_name + '/'
                if rval:
                    cv2.imwrite(pic_path + str(c) + '.jpg', frame)  # video_order.png
                    cv2.waitKey(1)
                else:
                    break
            vc.release()
        print("Finish splitting all videos")


    # split video into frames     
    get_frame_from_video(os.path.join(current_path, "video3"))

elif task

    path = os.path.join(current_path, "video2/")

    def make_gt(keys):
        for k in keys:
            v = keys[k]
            np_label = np.zeros(int(v[0]), dtype=np.int)
            
            if len(v) > 1:
                v = v[1:]
                assert len(v) % 2 == 0, "must be even"
                for i, f in enumerate(v):
                    if i % 2 == 1:
                        continue

                    s_ = v[i]
                    e_ = v[i+1]
                    np_label[int(s_): int(e_)+1] = 1

            np.save(os.path.join(current_path, 'label', k+".npy"), np_label)


    k = {} 
    # num of frames, ANOMALYS: start_, end_, start2_, end2_, .....
    k['Fire_0004'] = [420, 101, 195, 370, 413]            
    k['Fire_0005'] = [970, 47, 190, 312, 443, 779, 926]  
    k['Normal_0001'] = [550]  

    make_gt(k)

