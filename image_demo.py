import cv2
import time
import argparse
import os
import torch

import posenet
import random
import numpy as np
from metrics import RHand,LHand
"""
PoseNet
0: nose
1: right eye
2: left eye
3: right ear
4: left ear
5: right shoulder
6: left shoulder
7: right elbow
8: left elbow
9: right wrist
10: left wrist
11: right hip
12: left hip
13: right knee
14: left knee
15: right ankle
16: left ankle
"""

# time_recorder = []


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

def img_resize_scale(img, scale):    
    shape = np.shape(img)
    # scale0 = target_size/shape[0]
    # scale1 = target_size/shape[1]
    img = cv2.resize(img, (int(shape[1]*scale),int(shape[0]*scale)))
    return img

def main():
    #========================== my variables ==============================
    tracker = None
    detector_active = True    
    right_elbow, right_shoulder, right_wrist = 0,0,0
    #========================== my variables ==============================

    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    cap = cv2.VideoCapture('/home/mgharasu/Videos/Wo19.avi')#'/home/mgharasu/Documents/Noah_project/GT_creator/Wo14.MOV')

    start = time.time()
    # for f in filenames:
    colors = []
    for i in range(len(posenet.PART_NAMES)):
        colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    frame_counter  = 0
    t1 = time.time()
    while True:
        RHand.ts = time.time()
        LHand.ts = time.time()
        ret, frame = cap.read()
        
        if not ret:
            break
        frame_counter += 1
        img =  cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        # img = img_resize_scale(img,.6)      
        
        
        def detector(img):
            input_image, draw_image, output_scale = posenet.my_processImage(img, args.scale_factor, output_stride=output_stride)
            # input_image, draw_image, output_scale = posenet.read_imgfile(
            #     f, scale_factor=args.scale_factor, output_stride=output_stride)

            with torch.no_grad():
                input_image = torch.Tensor(input_image).cuda()

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.25)

            keypoint_coords *= output_scale # Get Keypoints
            return draw_image,keypoint_coords, keypoint_scores, pose_scores
        
        if detector_active:
            draw_image, keypoint_coords, keypoint_scores, pose_scores = detector(img)
            # detector_active = False

            if args.output_dir:
                draw_image, right_elbow, right_shoulder, right_wrist = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.0, min_part_score=0.0)
                
                # determining the bbox of wrist and keeping other points fix because they are fix, no important motion.
                #right_wrist = [x,y]
                x,y = right_wrist[0], right_wrist[1]
                l,t,b,r = y-40,x-40,y+40,x+40
        
        # 1- Adding to tracker
        # if tracker == None:
        #     tracker = C_TRACKER("CSRT")
            # tracker = 
        
        RHand.time_recorder.append(time.time()-RHand.ts)
        LHand.time_recorder.append(time.time()-LHand.ts)
        


        # 2- operations of rep count, velocity, duration and range of motion
        


            
           
        # if not args.notxt:
        #     print()
        #     print("Results for image: %s" % f)
        #     for pi in range(len(pose_scores)):
        #         if pose_scores[pi] == 0.:
        #             break
        #         print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
        #         for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
        #             print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
    seconds = time.time()
    end = seconds - t1
    print('Average FPS:', frame_counter/end)#len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
