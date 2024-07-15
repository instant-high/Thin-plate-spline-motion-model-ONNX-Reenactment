import cv2
import numpy as np


video_path = 'path_to_video.mp4'
cap = cv2.VideoCapture(video_path)

# Load the dictionary containing keypoints
kp_driving_data = np.load('kp_driving_all_frames.npz')

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    kp_driving = kp_driving_data[str(frame_idx)]


    frame_idx += 1

cap.release()
