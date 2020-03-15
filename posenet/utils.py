import math
import cv2
import numpy as np
import random
# from new_vars import index, gt_bicept_curl
import time
from metrics import RHand, LHand
import posenet.constants
from kmeans import km_main
from tracking import C_TRACKER
# from image_demo import 

# global RHand, LHand

# RHand.ts = time.time()
# LHand.ts = time.time()


class Rep: 
    def __init__(self):
        this.top = 0 
        this.bottom = [] 
        this.uROM = 0 
        this.dROM = 0 
        this.uDur = 0 
        this.dDur = 0 
        this.uVel = 0 
        this.dVel = 0


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()

    ### Rotates Loaded Video 
    img = cv2.resize(img, (1000, 640), interpolation = cv2.INTER_AREA)
    cv2.transpose(img, img)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)

def my_processImage(img, scale_factor=1.0, output_stride=16):
    img =  cv2.rotate(img,-cv2.ROTATE_90_CLOCKWISE)
    img=cv2.resize(img,(int(img.shape[1]*.5),int(img.shape[0]*.5)))
    return _process_input(img, scale_factor, output_stride)

def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img

def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def angle_between_points( p0, p1, p2 ):
  a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
  b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
  c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
  return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi


def angle_between_points( p0, p1, p2 ):
  a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
  b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
  c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
  a= a if a>0 else 1.0
  b= b if b>0 else 1.0
  c= c if c>0 else 1.0  
  return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi

def cluster(hands): 
    if len(hands.angle)>2:
        X=np.array(hands.angle).reshape((-1,1))
        km =km_main(X,2)
        # print(km,"-",max_c-min_c,"\n")
        # return False
        if km[0]<hands.min_c and km[0]!=0:
            hands.min_c = km[0]            
            # index_min = index
        elif km[1]<hands.min_c and km[1]!=0:
            hands.min_c=km[1]
            # index_min = index
        
        if km[0]>hands.max_c and km[0]!=0:
            hands.max_c=km[0]
            # index_max=index
        elif km[1]>hands.max_c and km[1]!=0:
            hands.max_c=km[1]


        if (hands.max_c-hands.min_c)>90:            
            return True
        
    return False
            

            # print("angle",angle)
            # return True

def fit_model_for_rep_count( hands, L_or_R):      
    
    # print(right_wrist[-1])
    if L_or_R=='R':
        hands.wrist.append(hands.points[-1][9].pt[1])
        ang = angle_between_points(hands.points[-1][5].pt, hands.points[-1][7].pt,hands.points[-1][9].pt)
    else:
        hands.wrist.append(hands.points[-1][10].pt[1])
        ang = angle_between_points(hands.points[-1][6].pt, hands.points[-1][8].pt,hands.points[-1][10].pt)

    # print(ang,",")
    hands.angle.append(ang)
  
    return cluster(hands)

# def detection_rep(rhand, lhand):

def peak_finder(sub_list, time_recorder, mode='top'):
    if mode =='top':
        value = min(sub_list)
    else:
        value = max(sub_list)

    idx=np.where(sub_list == value)
    # t2 = idx[0][0]/len(sub_list)*duration
    i = len(time_recorder)-len(sub_list)+idx[0][0]
    t2 = np.sum(time_recorder[:i])
    return value, t2, i#dx[0][0]    

def rep_detect(rhand, lhand):
    # peaks.append([index_min,index_max])
    # print(index_min,index_max)
    rhand.half_rep_count += 1
    lhand.half_rep_count += 1

    rhand.reps += 1 if rhand.half_rep_count%2==0 else 0
    lhand.reps += 1 if rhand.half_rep_count%2==0 else 0
                  
    # r_tc, r_duration = getTime(rhand.ts)
    # l_tc, l_duration = r_tc, r_duration

    if rhand.angle[-1]>=90 and lhand.angle[-1]>=90:
        # motion: down to top
        sub_list = np.array(rhand.wrist[rhand.index-len(rhand.angle):])
        rhand.top, r_duration, r_idx = peak_finder(sub_list, rhand.time_recorder, 'top')

        sub_list = np.array(lhand.wrist[lhand.index-len(lhand.angle):])
        lhand.top, l_duration, l_idx = peak_finder(sub_list, lhand.time_recorder, 'top')
    
        if rhand.half_rep_count>1 and lhand.half_rep_count>1:#rhand.top!=0 and lhand.top != 0:
            rhand.romDwnUp.append(abs(rhand.top-rhand.down))
            lhand.romDwnUp.append(abs(lhand.top-lhand.down))
    
            rhand.DDwnUp.append(r_duration)
            lhand.DDwnUp.append(l_duration)
    
            rhand.VDwnUp.append(rhand.romDwnUp[-1]/r_duration)
            lhand.VDwnUp.append(lhand.romDwnUp[-1]/l_duration)
        
    else:
        # motion: up to down
        sub_list = np.array(rhand.wrist[rhand.index-len(rhand.angle):])
        rhand.down, r_duration, r_idx = peak_finder(sub_list,rhand.time_recorder,'down')
        
        sub_list = np.array(lhand.wrist[lhand.index-len(lhand.angle):])
        lhand.down, l_duration, l_idx = peak_finder(sub_list,lhand.time_recorder,'down')
        
    
        if rhand.half_rep_count>1 and lhand.half_rep_count>1:#rhand.down != 0 and lhand.down!=0:
    
            rhand.romUpDwn.append(abs(rhand.top-rhand.down))
            lhand.romUpDwn.append(abs(lhand.top-lhand.down))
    
            rhand.DUpDwn.append(r_duration)
            lhand.DUpDwn.append(l_duration)
    
            rhand.VUpDwn.append(rhand.romUpDwn[-1]/r_duration)
            lhand.VUpDwn.append(lhand.romUpDwn[-1]/r_duration)
            
            # print(duration)        


    # rhand.ts = time.time()    
    rhand.max_c=0
    rhand.min_c=1000
    rhand.angle=[]
    rhand.time_recorder = rhand.time_recorder[r_idx:]

    # lhand.ts = time.time()    
    lhand.max_c=0
    lhand.min_c=1000
    lhand.angle=[]
    lhand.time_recorder = lhand.time_recorder[l_idx:]

def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):

    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []                                                 
    for ii, score in enumerate(instance_scores):
        if score < 0.0:#min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < 0.0:#min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    # def Start_process(img,cv_keypoints,)
    #"###################### added code #######################"
    # global RHand, LHand
    LHand.index += 1
    RHand.index += 1
    # print("index = ", str(index))
    if RHand.index == 1:
        RHand.points.append(np.array(cv_keypoints))        
        LHand.points.append(np.array(cv_keypoints)) 
    RHand.points.append(np.array(cv_keypoints))
    LHand.points.append(np.array(cv_keypoints))

    if RHand.index >= 1:
        rrep_found = fit_model_for_rep_count(RHand,'R')
        lrep_found = fit_model_for_rep_count(LHand,'L')

        if rrep_found and lrep_found:
            rep_detect(RHand,LHand)
    
    
    label={
        '0': 'nose',
        '1': 'right eye',
        '2': 'left eye',
        '3': 'right ear',
        '4': 'left ear',
        '5': 'right shoulder',
        '6': 'left shoulder',
        '7': 'right elbow',
        '8': 'left elbow',
        '9': 'right wrist',
        '10': 'left wrist',
        '11': 'right hip',
        '12': 'left hip',
        '13': 'right knee',
        '14': 'left knee',
        '15': 'right ankle',
        '16': 'left ankle'}
    
    right_elbow = [int(cv_keypoints[7].pt[0]), int(cv_keypoints[7].pt[1])]
    right_shoulder = [int(cv_keypoints[5].pt[0]), int(cv_keypoints[5].pt[1])]
    right_wrist = [int(cv_keypoints[9].pt[0]), int(cv_keypoints[9].pt[1])]

    left_elbow = [int(cv_keypoints[8].pt[0]), int(cv_keypoints[8].pt[1])]
    left_shoulder = [int(cv_keypoints[6].pt[0]), int(cv_keypoints[6].pt[1])]
    left_wrist = [int(cv_keypoints[10].pt[0]), int(cv_keypoints[10].pt[1])]

    angle=angle_between_points(right_wrist, right_elbow, right_shoulder)
    # global angles
    # print("angle:",angle)
    img = cv2.line(img, (right_shoulder[0],right_shoulder[1]), (right_elbow[0],right_elbow[1]), (0,0,255),2)
    img = cv2.line(img, (right_wrist[0],right_wrist[1]), (right_elbow[0],right_elbow[1]),(0,150,255), 2)

    img = cv2.line(img, (left_shoulder[0],left_shoulder[1]), (left_elbow[0],left_elbow[1]), (0,0,255),2)
    img = cv2.line(img, (left_wrist[0],left_wrist[1]), (left_elbow[0],left_elbow[1]),(0,150,255), 2)

    img=cv2.putText(img, str(RHand.index),(10,210),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    img=cv2.putText(img, 'Reps:'+str(RHand.reps),(10,230),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    if len(RHand.romUpDwn)>0:
        img=cv2.putText(img, "Range_UpDwn:"+str(RHand.romUpDwn[-1]),(10,40),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(RHand.romDwnUp)>0:
        img=cv2.putText(img, "Range_DwnUp:"+str(RHand.romDwnUp[-1]),(10,60),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)    
    img=cv2.putText(img, "================== Duration =====================",(10,80),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)    
    if len(RHand.DDwnUp)>0:
        img=cv2.putText(img, "Duration_DwnUp:"+str(RHand.DDwnUp[-1]),(10,100),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(RHand.DUpDwn)>0:
        img=cv2.putText(img, "Duration_UpDwn:"+str(RHand.DUpDwn[-1]),(10,120),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)    
    img=cv2.putText(img, "================== Velocity =====================",(10,140),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)        
    if len(RHand.VDwnUp)>0:
        img=cv2.putText(img, "Velocity_DwnUp:"+str(RHand.VDwnUp[-1]),(10,160),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(RHand.VUpDwn)>0:
        img=cv2.putText(img, "Velocity_UpDwn:"+str(RHand.VUpDwn[-1]),(10,180),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)

    for i,kp in enumerate(cv_keypoints):
        if i<17:
            colors=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
            point = (int(kp.pt[0]), int(kp.pt[1]))
            out_img=cv2.circle(out_img,point,2,colors,2)
            # out_img = cv2.putText(out_img,label[str(i)],point, cv2.FONT_HERSHEY_COMPLEX,0.5,colors,1)
            # outputVid.write(out_img)
    cv2.imshow('out',out_img)
    cv2.waitKey(1)
    if cv_keypoints:
        out_img = cv2.drawKeypoints(out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#<<<<<<< HEAD

    # out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
#=======
    #out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    
    #if len(cv_keypoints) > 9:
    #    print (cv_keypoints[9].pt[1])
    #    cv2.circle(out_img, (int(cv_keypoints[9].pt[0]), int(cv_keypoints[9].pt[1])), 2, (100, 255, 0), 2)
#>>>>>>> d32c96ecb457fc596829cb7abcc5c0738e20a294
    return out_img, right_elbow, right_shoulder, right_wrist

