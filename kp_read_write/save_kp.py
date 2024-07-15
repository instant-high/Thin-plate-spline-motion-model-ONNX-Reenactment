import cv2
import numpy as np


video_path = 'path_to_video.mp4'
cap = cv2.VideoCapture(video_path)

frame_idx = 0
kp_driving_dict = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    driving = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    driving = cv2.resize(driving, (256, 256)) / 255
    driving = np.transpose(driving[np.newaxis].astype(np.float32), (0, 3, 1, 2))

    ort_inputs = {kp_detector.get_inputs()[0].name: driving}
    kp_driving = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]

    kp_driving_dict[str(frame_idx)] = kp_driving

    frame_idx += 1

cap.release()

# Save the dictionary to a single .npz file
np.savez('kp_driving_all_frames.npz', **kp_driving_dict)
