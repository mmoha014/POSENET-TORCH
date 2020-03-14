import math
import cv2
import numpy as np
import random
# from new_vars import index, gt_bicept_curl
import time
from image_demo import index, points, angle, max_c, min_c,index_min,index_max, peaks, right_wrist, top, down, romUpDwn, romDwnUp, reps, half_rep_count,VDwnUp,VUpDwn,DDwnUp, DUpDwn, ts, time_recorder
import posenet.constants
from kmeans import km_main

global ts
ts = time.time()
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw


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

def cluster(ang): 
    global max_c, min_c, index_min,index_max, index , peaks, right_wrist, top, down, romUpDwn, romDwnUp, reps, half_rep_count,VDwn,VUp, DDwn, DUp, ts
    if len(ang)>2:
        X=np.array(ang).reshape((-1,1))
        km =km_main(X,2)
        # print(km,"-",max_c-min_c,"\n")
        # return False
        if km[0]<min_c and km[0]!=0:
            min_c = km[0]            
            # index_min = index
        elif km[1]<min_c and km[1]!=0:
            min_c=km[1]
            # index_min = index
        
        if km[0]>max_c and km[0]!=0:
            max_c=km[0]
            # index_max=index
        elif km[1]>max_c and km[1]!=0:
            max_c=km[1]
            # index_max=index
        
        global angle
        # if abs(km[0]-km[1])>100:
        #     # print("cut")           
        #     angle=[]
        #     max_c=0
        #     min_c=1000

        if (max_c-min_c)>60:            
            # peaks.append([index_min,index_max])
            # print(index_min,index_max)
            half_rep_count += 1
            reps += 1 if half_rep_count%2==0 else 0
            tc, duration = getTime(ts)
            if angle[-1]>=90:
                # motion: up to down
                top = min(right_wrist[index-len(angle):])
                if down != 0:
                    romDwnUp.append(abs(top-down))
                    DDwnUp.append(duration)
                    VDwnUp.append(romDwnUp[-1]/duration)
                    # print(duration)
            else:
                # motion: down to top
                down = max(right_wrist[index-len(angle):])
                if top!=0:
                    romUpDwn.append(abs(top-down))
                    DUpDwn.append(duration)
                    VUpDwn.append(romUpDwn[-1]/duration)
                    # print(duration)
            ts = tc    
            max_c=0
            min_c=1000
            angle=[]
            

            # print("angle",angle)
            # return True

def fit_model_for_rep_count(data):  
    xrw = [] #x_right_wrist
    yrw = [] #y_right_wrist
    xlw = [] #x_left_wrist
    ylw = [] #y_left_wrist
    cut_data = data[-1]
    x=[]
    y=[]
    # for i in range(len(data)):
    #     x.append(data[i][9][0])
    #     y.append(data[i][9][1])

    # for i in range(np.shape(cut_data)[0]):
    #     xrw.append(np.array(cut_data[i][9][0])) # right wrist
    #     yrw.append(np.array(cut_data[i][9][1]))
    #     xlw.append(np.array(cut_data[i][10][0]))
    #     ylw.append(np.array(cut_data[i][10][1]))

    # print(np.var([xrw,yrw]))
    #print(getAngle(cut_data[0][5], cut_data[0][7],cut_data[0][9])) # order of lines is important->change a and b makes the angle terrible (no change alot)
    
    global angle, right_wrist
    # cv_keypoints[7].pt[0]
    right_wrist.append(data[-1][9].pt[1])#data[-1][9][1])
    # print(right_wrist[-1])
    ang = angle_between_points(data[-1][5].pt, data[-1][7].pt,data[-1][9].pt)
    # print(ang,",")
    angle.append(ang)
    # if (len(angle))>30:
    #     xp = np.array(angle[-30:]).reshape((-1,1))
    #     yp = np.arange(len(xp))
    #     _x,model = myfit(xp,yp)
    #     print(model.coef_)
    #     a=0
    
    # a=copy.deepcopy(angle)
    
    cluster(angle)

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
    
    #"###################### added code #######################"
    global index, points, index, reps, romUpDwn, romDwnUp
    # index += 1
    # global index, points
    index += 1
    # print("index = ", str(index))
    if index == 1:
        points.append(np.array(cv_keypoints))
    points.append(np.array(cv_keypoints))
    if index >= 1:
        fit_model_for_rep_count(points)
    
    


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
    angle=angle_between_points(right_wrist, right_elbow, right_shoulder)
    # global angles
    # print("angle:",angle)
    out_img = cv2.line(out_img, (right_shoulder[0],right_shoulder[1]), (right_elbow[0],right_elbow[1]), (0,0,255),2)
    out_img = cv2.line(out_img, (right_wrist[0],right_wrist[1]), (right_elbow[0],right_elbow[1]),(0,150,255), 2)
    img=cv2.putText(img, str(index),(60,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    img=cv2.putText(img, 'Reps:'+str(reps),(60,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    if len(romUpDwn)>0:
        img=cv2.putText(img, "Range_UpDwn:"+str(romUpDwn[-1]),(10,140),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(romDwnUp)>0:
        img=cv2.putText(img, "Range_DwnUp:"+str(romDwnUp[-1]),(10,160),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(DUpDwn)>0:
        img=cv2.putText(img, "Duration_UpDwn:"+str(DUpDwn[-1]),(10,180),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(DDwnUp)>0:
        img=cv2.putText(img, "Duration_DwnUp:"+str(DDwnUp[-1]),(10,200),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(VDwnUp)>0:
        img=cv2.putText(img, "Velocity_DwnUp:"+str(VDwnUp[-1]),(10,220),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
    if len(VUpDwn)>0:
        img=cv2.putText(img, "Velocity_UpDwn:"+str(VUpDwn[-1]),(10,240),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)

    for i,kp in enumerate(cv_keypoints):
        if i<17:
            colors=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
            point = (int(kp.pt[0]), int(kp.pt[1]))
            out_img=cv2.circle(out_img,point,2,colors,2)
            # out_img = cv2.putText(out_img,label[str(i)],point, cv2.FONT_HERSHEY_COMPLEX,0.5,colors,1)
            cv2.imshow('out',out_img)
            cv2.waitKey(1)
    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
<<<<<<< HEAD

    # out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
=======
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    
    if len(cv_keypoints) > 9:
        print (cv_keypoints[9].pt[1])
        cv2.circle(out_img, (int(cv_keypoints[9].pt[0]), int(cv_keypoints[9].pt[1])), 2, (100, 255, 0), 2)
>>>>>>> d32c96ecb457fc596829cb7abcc5c0738e20a294
    return out_img

