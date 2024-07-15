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

parser.add_argument("--crop_scale", type=float, default=1.2, help='Bbox size around detected head')
parser.add_argument("--face_parser", action="store_true", help="Use face_parsing mask")
parser.add_argument("--parser_index", default="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17", type=lambda x: list(map(int, x.split(','))),help='index of NOT masked parts')
parser.add_argument("--animation_factor", type=float, default=1, help='Animation strength')

parser.add_argument("--enhancement", default='none', choices=['none', 'gpen'])  #, 'gfpgan', 'codeformer', 'restoreformer'])
parser.add_argument('--blending', default=5, type=float, help='Amount of face enhancement blending 1 - 10')

parser.add_argument("--audio", dest="audio", action="store_true", help="Keep audio")

args = parser.parse_args()


device = 'cuda'

# face detector model:      
detector = RetinaFace("utils/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)

# kp_detector and tpsmm models:
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}),"CPUExecutionProvider"] # EXHAUSTIVE
      
kp_detector = onnxruntime.InferenceSession('tpsmm/kp_detector.onnx', sess_options=session_options, providers=providers)    
tpsm_model = onnxruntime.InferenceSession('tpsmm/tpsmm_rel.onnx', sess_options=session_options, providers=providers)    

# face parser model:
if args.face_parser:
    from face_parser.face_parser import FACE_PARSER
    facemask = FACE_PARSER(model_path="face_parser/face_parser.onnx",device=device)
    parser_index = args.parser_index
    assert type(parser_index) == list

# face enhancer:
if args.enhancement == 'gpen':
    from GPEN.GPEN import GPEN
    gpen256 = GPEN(model_path="GPEN/GPEN-BFR-256.onnx", device="cuda")
        
def process_image(model, img, size, crop_scale=1.0):
    try:
        bboxes, kpss = model.detect(img,(256, 256), det_thresh=0.6)
        if len(kpss) == 0:
            raise ValueError("No face detected")
        
        aimg, mat = get_cropped_head_256(img, kpss[0], size=size, scale=args.crop_scale)
        return aimg, mat
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def get_driving_bbox(model, img, size, crop_scale=1.0):
    image_height, image_width = img.shape[:2]

    try:
        bboxes, kpss = model.detect(img, (256, 256), det_thresh=0.6)
        
        if len(kpss) == 0:
            raise ValueError("No bounding box detected")
        
        bbox = bboxes[0][:-1]
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        new_x1 = max(0, x1 - width / 2)
        new_y1 = max(0, y1 - height / 2)
        new_x2 = min(image_width, x2 + width / 2)
        new_y2 = min(image_height, y2 + height / 2)
        extended_bbox = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
        
        return extended_bbox
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

        
def relative_kp(kp_target, kp_driving, kp_driving_initial):
    source_area = ConvexHull(kp_target[0]).volume
    driving_area = ConvexHull(kp_driving_initial[0]).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_value_diff = (kp_driving - kp_driving_initial)
    kp_value_diff *= adapt_movement_scale * args.animation_factor
    kp_new = kp_value_diff + kp_target

    return kp_new


def reenact():
    
    # enhancer blending
    blend = args.blending/10
    
    # static mask:
    r_mask = cv2.imread('mask.jpg')
    r_mask = r_mask / 255

    # load driving video:
    cap_driving = cv2.VideoCapture(args.driving)
    fps_driving = cap_driving.get(cv2.CAP_PROP_FPS)
    total_frames_driving = int(cap_driving.get(cv2.CAP_PROP_FRAME_COUNT))
    w_driving = int(cap_driving.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_driving = int(cap_driving.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret_d, frame_driving = cap_driving.read()
    
    #- different options for driving video first frame: -# 
    
    # pre-recorded *.npz -  
    
    # pre-cropped:
    #face_d = frame_driving

    # just detection, no alignment:
    #coords = get_driving_bbox(detector, frame_driving, 256, crop_scale=args.crop_scale)
    
    # detection and alignment:
    #face_d, matrix = process_image(detector, frame_driving, 256, crop_scale=args.crop_scale)

    
    # cropping for static crop or detection only:    
    coords = get_driving_bbox(detector, frame_driving, 256, crop_scale=args.crop_scale)
    face_d = frame_driving[coords[1]:coords[3], coords[0]:coords[2]]
    #
    
    
    # kp_driving_initial:    
    face_d = cv2.cvtColor(face_d, cv2.COLOR_RGB2BGR)    
    face_d = cv2.resize(face_d, (256, 256))/ 255
    face_d = np.transpose(face_d[np.newaxis].astype(np.float32), (0, 3, 1, 2))
    ort_inputs = {kp_detector.get_inputs()[0].name: face_d}
    kp_driving_initial = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]


    # load target video:
    cap_target = cv2.VideoCapture(args.target)
    fps_target = cap_target.get(cv2.CAP_PROP_FPS)
    total_frames_target = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))
    w_target = int(cap_target.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_target = int(cap_target.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # first frame target (source image)
    ret_t, frame_target = cap_target.read()
    face, matrix = process_image(detector, frame_target, 256, crop_scale=args.crop_scale)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255
    face = np.transpose(face[np.newaxis].astype(np.float32), (0, 3, 1, 2))
    ort_inputs = {kp_detector.get_inputs()[0].name: face}
    kp_target = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]
    
    if args.audio:
        writer = cv2.VideoWriter(('_temp.mp4'),cv2.VideoWriter_fourcc('m','p','4','v'), fps_target, (w_target, h_target))
    else:
        writer = cv2.VideoWriter((args.output),cv2.VideoWriter_fourcc('m','p','4','v'), fps_target, (w_target, h_target))
    
    total_frames = min(total_frames_driving, total_frames_target)
    
    #cap_target.set(1,0)
    #cap_driving.set(1,0)
               
    for index in tqdm(range(total_frames)):
        ret_d, frame_driving = cap_driving.read()
        ret_t, frame_target = cap_target.read()
        
        if not ret_d:break
        if not ret_t:break
        
        #- different options for driving video each frame: -#
        
        # pre-recorded *.npz -
        
        # pre-cropped:    
        #driving = frame_driving
            
        # detection and alignment:
        #driving, matrix_ = process_image(detector, frame_driving, 256, crop_scale=args.crop_scale)
        
        # detection, no alignment:
        #coords = get_driving_bbox(detector, frame_driving, 256, crop_scale=args.crop_scale)
        
        # cropping for static crop or detection only: 
        driving = frame_driving[coords[1]:coords[3], coords[0]:coords[2]]
        #
  
        driving = cv2.cvtColor(driving, cv2.COLOR_RGB2BGR)

        driving = cv2.resize(driving, (256, 256))/ 255
        driving = np.transpose(driving[np.newaxis].astype(np.float32), (0, 3, 1, 2))
        ort_inputs = {kp_detector.get_inputs()[0].name: driving}
        kp_driving = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0] 
            
        # target:            
        target, t_matrix = process_image(detector, frame_target, 256, crop_scale=args.crop_scale)
        inverse_matrix = cv2.invertAffineTransform(t_matrix)   
        
        target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        target = cv2.resize(target, (256, 256))/ 255
        target = np.transpose(target[np.newaxis].astype(np.float32), (0, 3, 1, 2))
        ort_inputs = {kp_detector.get_inputs()[0].name: target}
        kp_target = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]           
        kp_norm = relative_kp(kp_target=kp_target, kp_driving=kp_driving,kp_driving_initial=kp_driving_initial)
            
        ort_inputs = {tpsm_model.get_inputs()[0].name: kp_target,tpsm_model.get_inputs()[1].name: target, tpsm_model.get_inputs()[2].name: kp_norm, tpsm_model.get_inputs()[3].name: driving}
        out = tpsm_model.run([tpsm_model.get_outputs()[0].name], ort_inputs)[0]
            
        result = np.transpose(out.squeeze(), (1, 2, 0))
        result = cv2.cvtColor(result*255, cv2.COLOR_RGB2BGR)
        result_copy = result.copy()
                
        if args.face_parser:
            p_mask = facemask.create_region_mask(result.astype(np.uint8), parser_index)
            p_mask = cv2.resize(p_mask,(256, 256))
            p_mask = cv2.cvtColor(p_mask, cv2.COLOR_GRAY2RGB)       
            p_mask = cv2.rectangle(p_mask, (5, 5), (251, 251), (0, 0, 0), 15)
            p_mask = cv2.GaussianBlur(p_mask,(11, 11),cv2.BORDER_DEFAULT)
            mask = cv2.warpAffine(p_mask, inverse_matrix, (w_target, h_target))
        else:
            mask = cv2.warpAffine(r_mask, inverse_matrix, (w_target, h_target))

        if args.enhancement == 'gpen':
            result = gpen256.enhance(result)
            result = cv2.addWeighted(result.astype(np.float32), blend, result_copy.astype(np.float32), 1.-blend, 0.0)

        
        result = cv2.warpAffine(result, inverse_matrix, (w_target, h_target))

        img = (mask * result + (1 - mask) * frame_target).astype(np.uint8)           

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
    
reenact()

if args.audio:
    print ("Writing Audio...")
    # driving or target audio:
    #command = 'ffmpeg.exe -y -vn -i ' + '"' + args.driving + '"' + ' -an -i ' + '_temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + args.output + '"'
    command = 'ffmpeg.exe -y -vn -i ' + '"' + args.target + '"' + ' -an -i ' + '_temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + args.output + '"'
    
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.system('cls')

    if os.path.exists('_temp.mp4'):
        os.remove('_temp.mp4')

