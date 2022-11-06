import os
from imutils.video import FPS
import cv2
import math

def video_to_frames(input_loc, output_loc): 
    path=os.listdir(input_loc)
    count=1
    for i in path:
        try:
            main_folder=os.path.join(input_loc,i)
            for folder in os.listdir(main_folder):
                print(folder)
                video_path=os.path.join(main_folder,folder)
                for videos in os.listdir(video_path):
                    if videos.endswith(('.mp4','.avi','.mkv','.AVI','.MP4','.MKV')):
                        count+=1
                        if count%10==0: print(count)
                        # one frame per second
                        videoFile = os.path.join(video_path,videos)
                        cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
                        frameRate = cap.get(5) #frame rate
                        x=1
                        while(cap.isOpened()):
                            frameId = cap.get(1)
                            ret, frame = cap.read()
                            if (ret != True):
                                break
                            if (frameId % math.floor(frameRate) == 0):
                                filename =output_loc+"/"+videos+"_"+str(int(frameId))+".jpg"
                                cv2.imwrite(filename, frame)

        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    video_loc = "videos"
    images_loc = "images"
    video_to_frames(video_loc, images_loc)  