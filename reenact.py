import os
import subprocess
import platform
import cv2
import argparse
import numpy as np
from scipy.spatial import ConvexHull

import onnxruntime
onnxruntime.set_default_logger_severity(3) # gpen

from tqdm import tqdm
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256


parser = argparse.ArgumentParser()
parser.add_argument("--driving", type=str, default='driving.mp4')
parser.add_argument("--target", type=str, default='target.mp4')
parser.add_argument("--output", type=str, default='output.mp4')

parser.add_argument("--target_audio", action="store_true", help="Use original audio")
parser.add_argument("--driving_audio", action="store_true", help="Use driving video audio")

parser.add_argument("--scale", default=10, help="Scale factor input video, 10 = 1")

# not yet implemented
#parser.add_argument("--startpos", dest="startpos", type=int, default=0, help="Frame to start from")
#parser.add_argument("--endpos", dest="endpos", type=int, default=0, help="Frame to end inference")
#parser.add_argument("--loop", action="store_true", help="Loop driving video")

# target video:
parser.add_argument("--crop_scale", type=float, default=1.15, help='bbox size around aligned head target video') #1.25

# driving video:
parser.add_argument("--tracking", action="store_true", help="Facetracking driving video") # not recommended

parser.add_argument("--animation_factor", type=float, default=1, help='Animation strength') 
parser.add_argument("--mouth", action="store_true", help="Try to close the mouth/keep pose of first driving frame")

parser.add_argument("--enhancement", default='none', choices=['none', 'gpen'])  #, 'gfpgan', 'codeformer', 'restoreformer'])
parser.add_argument("--blending", default=5, type=float, help='Amount of face enhancement blending 1 - 10')

args = parser.parse_args()

device = 'cuda'

# face detector model:      
detector = RetinaFace("utils/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)

# kp_detector and tpsmm model:
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}),"CPUExecutionProvider"] # EXHAUSTIVE
      
kp_detector = onnxruntime.InferenceSession('tpsmm/kp_detector.onnx', sess_options=session_options, providers=providers)    
tpsm_model = onnxruntime.InferenceSession('tpsmm/tpsmm_rel.onnx', sess_options=session_options, providers=providers)    

# face enhancer:
if args.enhancement == 'gpen':
    from GPEN.GPEN import GPEN
    gpen256 = GPEN(model_path="GPEN/GPEN-BFR-256.onnx", device="cuda")
    
        
def process_frame(model, img, size, crop_scale=1.0):
    try:
        bboxes, kpss = model.detect(img,(256, 256), det_thresh=0.6)
        if len(kpss) == 0:
            raise ValueError("No face detected")
        
        aimg, mat = get_cropped_head_256(img, kpss[0], size=size, scale=args.crop_scale)
        return aimg, mat
    
    except Exception as e:
        print(f"Error aligning image: {e}")
        return None, None


def get_driving_bbox(model, img):
    image_height, image_width = img.shape[:2]

    try:
        bboxes, kpss = model.detect(img, (256, 256), det_thresh=0.6)
        
        if len(bboxes) == 0:
            raise ValueError("No bounding box detected")
        
        bbox = bboxes[0][:-1]
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        new_x1 = max(0, x1 - width / 2)
        new_y1 = max(0, y1 - height / 4)
        new_x2 = min(image_width, x2 + width / 2)
        new_y2 = min(image_height, y2 + height / 4 )
        extended_bbox = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
        
        return extended_bbox
    
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None

    
        
def relative_kp_orig(kp_target, kp_driving, kp_driving_initial):

    target_area = ConvexHull(kp_target[0]).volume
    driving_area = ConvexHull(kp_driving_initial[0]).volume
    adapt_movement_scale = np.sqrt(target_area) / np.sqrt(driving_area)

    kp_value_diff = (kp_driving - kp_driving_initial)
    kp_value_diff *= adapt_movement_scale * args.animation_factor
    kp_new = kp_value_diff + kp_target

    return kp_new


def relative_kp(kp_target, kp_driving, kp_driving_initial):

    # Try to close the mouth/keep pose of first driving frame
    indices_mouth = [32, 10, 8] # [32, 10, 8, 17] # [ 32, 10, 8, 17, 23, 36]
    if args.mouth:
        kp_driving_initial[0, indices_mouth] = kp_target[0, indices_mouth]
        
    target_area = ConvexHull(kp_target[0]).volume
    driving_area = ConvexHull(kp_driving[0]).volume
    adapt_movement_scale = np.sqrt(target_area) / np.sqrt(driving_area)

    kp_value_diff = (kp_driving - kp_driving_initial)
    
    # do not move shoulders at image bottom 25, 27, 35, 37, 46, 47 / image top 26, 28, 29
    indices_shoulder = [25, 27, 35, 37, 46, 47, 26, 28, 29]
    kp_value_diff[0, indices_shoulder] = [0.0, 0.0]

    kp_value_diff *= adapt_movement_scale * args.animation_factor
    kp_new = kp_value_diff + kp_target
    
    return kp_new

    
def reenact():
    
    # enhancer blending:
    blend = args.blending/10

    # mask    
    r_mask = np.zeros((256, 256), dtype=np.uint8)
    for i in range(32):
        value = int((i / 32) * 255)
        r_mask = cv2.rectangle(r_mask, (i, i), (255-i, 255-i), value, -1)
    r_mask = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2RGB) / 255
    
    # load target video:
    cap_target = cv2.VideoCapture(args.target)
    fps_target = cap_target.get(cv2.CAP_PROP_FPS)
    total_frames_target = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))
    w_target = int(cap_target.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_target = int(cap_target.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # resize target input        
    scale  = int(args.scale)/10
    w_target = int(w_target*scale)
    h_target = int(h_target*scale)
    if w_target %2 !=0 : w_target = w_target - 1
    if h_target %2 !=0 : h_target = h_target - 1
    
    # load driving video first frame:
    cap_driving = cv2.VideoCapture(args.driving)
    fps_driving = cap_driving.get(cv2.CAP_PROP_FPS)
    
    # re-encode driving video anyway to get same framerate as target / keep driving audio in sync
    command = 'ffmpeg.exe -y -i ' + '"' + args.driving + '"' + ' -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -r ' + str(fps_target) +  ' "' + '_driving_temp.mp4' + '"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    args.driving = '_driving_temp.mp4'
    
    os.system('cls')
    print("Re-encoded driving video from " + str(fps_driving) + " fps to " + str(fps_target) +" fps to match target video framerate") 

    if args.target_audio and args.driving_audio:
        print ("Warning")
        print ("You can't select target audio and driving audio at the same time, default to driving audio")
        args.target_audio = False

    if args.target_audio or args.driving_audio:
        writer = cv2.VideoWriter(('_temp.mp4'),cv2.VideoWriter_fourcc('m','p','4','v'), fps_target, (w_target, h_target))
    else: # no audio
        writer = cv2.VideoWriter((args.output),cv2.VideoWriter_fourcc('m','p','4','v'), fps_target, (w_target, h_target))
        
            
    total_frames_driving = int(cap_driving.get(cv2.CAP_PROP_FRAME_COUNT))
    w_driving = int(cap_driving.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_driving = int(cap_driving.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret_d, frame_driving = cap_driving.read()
    
    # pre-recorded *.npz -  

    # cropping first driving frame for static crop or detection:    
    coords = get_driving_bbox(detector, frame_driving)
    face_d = frame_driving[coords[1]:coords[3], coords[0]:coords[2]]
    
    # get kp_driving_initial first frame:    
    face_d = cv2.cvtColor(face_d, cv2.COLOR_RGB2BGR)    
    face_d = cv2.resize(face_d, (256, 256))/ 255
    face_d = np.transpose(face_d[np.newaxis].astype(np.float32), (0, 3, 1, 2))
    ort_inputs = {kp_detector.get_inputs()[0].name: face_d}
    
    kp_driving_initial = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]
    
    total_frames = min(total_frames_driving, total_frames_target)
        
    cap_target.set(1,0)
    cap_driving.set(1,0)
               
    for index in tqdm(range(total_frames)):
      
        ret_t, frame_target = cap_target.read()
        frame_target = cv2.resize(frame_target,(w_target,h_target))
        
        ret_d, frame_driving = cap_driving.read()

         
        if not ret_d:break
        if not ret_t:break
        
        # pre-recorded *.npz kp from cropped video #
        
        # track driving video face:
        if args.tracking:
            coords = get_driving_bbox(detector, frame_driving)
            
        if coords == None:
            driving = frame_driving
        else:
            driving = frame_driving[coords[1]:coords[3], coords[0]:coords[2]]
        
        #driving_copy = driving.copy()
        #cv2.imshow("Driving",cv2.resize(driving,(256,256)))
        
        driving = cv2.cvtColor(driving, cv2.COLOR_RGB2BGR)        
        driving = cv2.resize(driving, (256, 256)) / 255
        driving = np.transpose(driving[np.newaxis].astype(np.float32), (0, 3, 1, 2))
        ort_inputs = {kp_detector.get_inputs()[0].name: driving}
        kp_driving = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]    
            
        # target:         
        target, t_matrix = process_frame(detector, frame_target, 256, crop_scale=args.crop_scale)
        inverse_matrix = cv2.invertAffineTransform(t_matrix)
        target_copy = target.copy()
        cv2.resize(target_copy,(256,256))
        # cv2.imshow("Target",target)
        target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        target = cv2.resize(target, (256, 256)) / 255
        target = np.transpose(target[np.newaxis].astype(np.float32), (0, 3, 1, 2))
        
        ort_inputs = {kp_detector.get_inputs()[0].name: target}
        kp_target = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]

        kp_norm = relative_kp(kp_target=kp_target, kp_driving=kp_driving,kp_driving_initial=kp_driving_initial)

        ort_inputs = {tpsm_model.get_inputs()[0].name: kp_target,tpsm_model.get_inputs()[1].name: target, tpsm_model.get_inputs()[2].name: kp_norm, tpsm_model.get_inputs()[3].name: driving}
        out = tpsm_model.run([tpsm_model.get_outputs()[0].name], ort_inputs)[0]
        
        result = np.transpose(out.squeeze(), (1, 2, 0))
        result = result * 255
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result_copy = result.copy()
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if args.enhancement == 'gpen':
            result = gpen256.enhance(result)
            result = cv2.addWeighted(result.astype(np.float32), blend, result_copy.astype(np.float32), 1.-blend, 0.0)
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        result = cv2.warpAffine(result, inverse_matrix, (w_target, h_target))
        mask = cv2.warpAffine(r_mask, inverse_matrix, (w_target, h_target))        
        img = (mask * result + (1 - mask) * frame_target).astype(np.uint8)           
        
        #img[0:128, 0:128] = cv2.resize(target_copy,(128,128))
        #img[0:128, 128:256] = cv2.resize(driving_copy,(128,128))

        writer.write(img)
        
        cv2.imshow("Press Esc to stop...",img) 
        k = cv2.waitKey(1)
        if k == 27:
            cap_driving.release()
            cap_target.release()        
            writer.release()
            cv2.destroyAllWindows()            
            break
        
    cap_driving.release()
    cap_target.release()
    writer.release()
    cv2.destroyAllWindows()

    if args.driving_audio:
        print ("Writing Audio...")
        # use driving video audio
        command = 'ffmpeg.exe -y -vn -i ' + '"' + args.driving + '"' + ' -an -i ' + '_temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + args.output + '"'
        subprocess.call(command, shell=platform.system() != 'Windows')
        os.system('cls')
    
        if os.path.exists('_temp.mp4'):
            os.remove('_temp.mp4')
            os.remove('_driving_temp.mp4')

    if args.target_audio:
        print ("Writing Audio...")
        # use target video audio
        ##command = 'ffmpeg.exe -y -vn -i ' + '"' + args.driving + '"' + ' -an -i ' + '_temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + args.output + '"'
        command = 'ffmpeg.exe -y -vn -i ' + '"' + args.target + '"' + ' -an -i ' + '_temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + args.output + '"'
        subprocess.call(command, shell=platform.system() != 'Windows')
        os.system('cls')
    
        if os.path.exists('_temp.mp4'):
            os.remove('_temp.mp4')
            os.remove('_driving_temp.mp4')    

reenact()


