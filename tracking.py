import cv2
from random import randint
from linear_assignment import linear_assignment
import numpy as np
# from utils import *
# import KalmanFilter as kf_tracker
# from setting import max_age
import sys
# from preprocessing import C_PREPROCESSING

def correct_position(box):    
    x = int(round(abs(box[0])))
    y = int(round(abs(box[1])))
    w = int(round(abs(box[2])))
    h = int(round(abs(box[3])))
    startX, startY, endX, endY = x,y, x+w, y+h
    return [startX, startY, endX, endY]

def box_iou2(a, b):
    # print("a:",str(a))
    # print("b:",str(b))
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''                
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    # print("w_insect: ", w_intsec, ", h_insect: ", h_intsec)
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
    # print("s_a:",str(s_a),", s_b: ",str(s_b))
    # print(str(s_intsec)+"/"+str(abs(s_a))+" + "+str(abs(s_b)+" - "+str(s_intsec)))
    # print("s_intesc:",s_intsec," s_a: ", s_a, "s_b: ",s_b)
    # print(str(float(s_intsec))+"/"+str((abs(s_a) + abs(s_b) -s_intsec)))
    # if s_intsec == 0:
    return float(s_intsec)/(abs(s_a) + abs(s_b) -s_intsec) if s_intsec!=0 else 0

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0


class C_TRACKER:
    
    def __init__(self, tracker_type):        
        self.__type = tracker_type        
        self.__trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'Kalman_Filter']
        self.colors = []
        self.__frame_size = None
        self.__predicted_boxes = []
        self.__identity = 0
        self.__bgs = None
        self.initialized = True
        
        if tracker_type is not "Kalman_Filter":
            self.__tracker = cv2.MultiTracker_create()

        else:
            self.__tracker = []
            
    def Empty(self):
        # print("Empty function:", self.initialized)
        # if self.initialized:
        #     self.initialized = False
        #     return True
        # else:
        #     return False
        # print("tracker.Empty(): ", True if len(self.__predicted_boxes)==0 else False)
        if len(self.__predicted_boxes)>0:
            return False
        return True

    def Get_trackerSize(self):
        return get_size(self.__tracker)+get_size(self.colors)+get_size(self.__predicted_boxes)

    def __createTrackerByName(self, TrackerType):
        # Create a tracker based on tracker name
        if TrackerType == self.__trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif TrackerType == self.__trackerTypes[1]: 
            tracker = cv2.TrackerMIL_create()
        elif TrackerType == self.__trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif TrackerType == self.__trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif TrackerType == self.__trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif TrackerType == self.__trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif TrackerType == self.__trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif TrackerType == self.__trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in self.__trackerTypes:
                print(t)
            
        return tracker

    # def Add_Tracker(self, frame, BoundingBoxes):
    #     for bb in BoundingBoxes:
    #         self.__tracker.add(self.__createTrackerByName(self.__type),frame,(bb[0],bb[1],bb[2],bb[3]))
    #         self.__colors.append((randint(0,255), randint(0,255), randint(0,255)))
    def Add_Tracker(self, frame, BoundingBoxes, is_switch=False):
        self.__frame_size = frame.shape[0]
        if is_switch:
            tmp_colors = self.colors
            self.colors = []
        # print("# objs: ", len(BoundingBoxes), " - is_switch: ", is_switch)
        for i, bb in enumerate(BoundingBoxes):
            # print("before: ",bb[0],",",bb[1],",",bb[2],",",bb[3])
            self.__tracker.add(self.__createTrackerByName(self.__type),frame,(bb[0],bb[1],bb[2],bb[3]))
            
            self.__predicted_boxes.append((bb[0],bb[1],bb[2],bb[3]))
            
            # self.__colors.append((randint(0,255), randint(0,255), randint(0,255)))            
            #--- version 2 -----
            if is_switch:
                # print("after", tmp_colors[i])
                self.colors.append((tmp_colors[i][0], tmp_colors[i][1], tmp_colors[i][2], tmp_colors[i][3], tmp_colors[i][4]))
                
            else:
                self.colors.append((randint(0,255), randint(0,255), randint(0,255), self.__Get_NewIdentity(),0))
                # print("new colors: ", self.colors[-1])

    def update(self, frame):#boxes, detected_boxes):
        if self.__type != 'Kalman_Filter':
            success, self.__predicted_boxes = self.__tracker.update(frame)
            for i, newbox in enumerate(self.__predicted_boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, self.colors[i][0:3], 2, 1)        
        
        return self.__predicted_boxes,frame
    
    '''    
    finding objects that are tracking
    assigning new tracker to new detected objects
    removing trackers whose objects are not in scene
    '''
    def update_pipeline(self, frame, detected_boxes):
        if self.__type != "Kalman_Filter":
            success, self.__predicted_boxes = self.__tracker.update(frame)
            self.__tracker, self.colors = self.__pipline_OpenCV(frame, detected_boxes, self.__predicted_boxes, self.__type, self.colors, self.__createTrackerByName )
        else:
            self.__tracker, self.colors = self.__pipeline_KalmanFilter(frame, detected_boxes, self.colors)
            return self.__tracker, self.colors
        # success, self.__predicted_boxes = self.__tracker.update(frame)
        # for i, newbox in enumerate(self.__predicted_boxes):
        #     p1 = (int(newbox[0]), int(newbox[1]))
        #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #     cv2.rectangle(frame, p1, p2, self.__colors[i], 2, 1)
        
        # return frame
    
    def Get_predicted_boxes(self):
        return self.__predicted_boxes
   

    def assign_detections_to_trackers(self, trackers, detections, iou_thrd = 0.3):
        '''
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.
        '''    
        
        IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        for t,trk in enumerate(trackers):
            #trk = convert_to_cv2bbox(trk) 
            for d,det in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
                IOU_mat[t,d] = box_iou2(trk,det) 
        
        # Produces matches       
        # Solve the maximizing the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)
        
        matched_idx = linear_assignment(-IOU_mat)        

        unmatched_trackers, unmatched_detections = [], []
        for t,trk in enumerate(trackers):
            if(t not in matched_idx[:,0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(detections):
            if(d not in matched_idx[:,1]):
                unmatched_detections.append(d)

        matches = []
    
        # For creating trackers we consider any detection with an 
        # overlap less than iou_thrd to signifiy the existence of 
        # an untracked object
        
        for m in matched_idx:
            if(IOU_mat[m[0],m[1]]<iou_thrd):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers) 


    # def __pipeline_KalmanFilter(self, frame, roi_boxes, colors):
    #     z_box = [correct_position(x) for x in roi_boxes]
    #         # x_box = []
    #     x_box = []#[correct_position(x) for x in tracker_list]
    #     if len(self.__tracker) > 0:
    #             for trk in self.__tracker:
    #                 x_box.append(trk.box)
        
    #     matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  

    #     # Deal with matched detections
    #     newcolors = []
    #     if matched.size >0:
    #         for trk_idx, det_idx in matched:
    #             z = z_box[det_idx]
    #             z = np.expand_dims(z, axis=0).T
    #             tmp_trk= self.__tracker[trk_idx]
    #             tmp_trk.kalman_filter(z)
    #             xx = tmp_trk.x_state.T[0].tolist()
    #             xx =[xx[0], xx[2], xx[4], xx[6]]
    #             x_box[trk_idx] = xx
    #             tmp_trk.box =xx
    #             tmp_trk.hits += 1
    #             tmp_trk.no_losses = 0
    #             newcolors.append(colors[trk_idx])
        
    #     # Deal with unmatched detections      
    #     if len(unmatched_dets)>0:
    #         for idx in unmatched_dets:
    #             z = z_box[idx]
    #             z = np.expand_dims(z, axis=0).T
    #             tmp_trk = kf_tracker.Tracker() # Create a new tracker
    #             x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
    #             tmp_trk.x_state = x
    #             tmp_trk.predict_only()
    #             xx = tmp_trk.x_state
    #             xx = xx.T[0].tolist()
    #             xx =[xx[0], xx[2], xx[4], xx[6]]
    #             tmp_trk.box = xx
    #             tmp_trk.id = self.__Get_NewIdentity()#.popleft() # assign an ID for the tracker
    #             # track_id_list+=1
    #             self.__tracker.append(tmp_trk)
    #             x_box.append(xx)
    #             newcolors.append((randint(0,255), randint(0,255),randint(0,255), tmp_trk.id, 0))
        
    #     # Deal with unmatched tracks       
    #     if len(unmatched_trks)>0:
    #         for trk_idx in unmatched_trks:
    #             tmp_trk = self.__tracker[trk_idx]
    #             tmp_trk.no_losses += 1
    #             tmp_trk.predict_only()
    #             xx = tmp_trk.x_state
    #             xx = xx.T[0].tolist()
    #             xx =[xx[0], xx[2], xx[4], xx[6]]
    #             tmp_trk.box =xx
    #             x_box[trk_idx] = xx
    #             newcolors.append(colors[trk_idx])
                                    
    #     # Book keeping
    #     # deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)          
    #     final_tracker_list = []
    #     final_color_list = []
    #     self.__predicted_boxes = []
    #     for i,x in enumerate(self.__tracker):
    #         if x.no_losses<=max_age:
    #             final_tracker_list.append(x)
    #             final_color_list.append(newcolors[i])
    #             self.__predicted_boxes.append(x.box)
    #     # tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
    #     return final_tracker_list, final_color_list 

    # def initialize_kalmanfilter_tracker(self):
    #     newcolors = []
    #     self.__tracker = []
    #     if len(self.__predicted_boxes)>0:
    #         for i,box in enumerate(self.__predicted_boxes):
    #             z = box 
    #             z = np.expand_dims(z, axis=0).T
    #             tmp_trk = kf_tracker.Tracker() # Create a new tracker
    #             x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
    #             tmp_trk.x_state = x
    #             tmp_trk.predict_only()
    #             xx = tmp_trk.x_state
    #             xx = xx.T[0].tolist()
    #             xx =[xx[0], xx[2], xx[4], xx[6]]
    #             tmp_trk.box = xx
    #             tmp_trk.id = self.colors[3]#.popleft() # assign an ID for the tracker
    #             # track_id_list+=1
    #             self.__tracker.append(tmp_trk)
    #             # x_box.append(xx)
    #             newcolors.append(self.colors[i])
    #     self.colors = newcolors

    
    # def isKalmanFilter(self):
    #     return self.__type == "Kalman_Filter"

    def __pipline_OpenCV(self, frame, roi_boxes, tracker_list, tracker_type, colors, function):            
            z_box = [correct_position(x) for x in roi_boxes]
            # x_box = []
            x_box = [correct_position(x) for x in tracker_list]

            # if len(tracker_list.getObjects()) > 0:
                # for trk in tracker_list.getObjects():
                    # x_box.append(trk.box)

            matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  

            newcolors = []
            multiTracker = cv2.MultiTracker_create() #add self.__predicted_boxes
            self.__predicted_boxes = []
            for m in matched:
                bb = roi_boxes[m[1]]
                self.__predicted_boxes.append(bb)
                multiTracker.add(function(tracker_type), frame, (bb[0],bb[1],bb[2],bb[3]))
                newcolors.append(colors[m[0]])

            for m in unmatched_dets:
                bb = roi_boxes[m]
                self.__predicted_boxes.append(bb)
                multiTracker.add(function(tracker_type), frame, (bb[0],bb[1],bb[2],bb[3]))
                newcolors.append((randint(0,255), randint(0,255),randint(0,255), self.__Get_NewIdentity(), 0))

            return multiTracker, newcolors

    def Get_MOTChallenge_Format(self):
        # print("in tracker class-> get_motchallenge_format function: ",self.__predicted_boxes)
        tmp = []
        try:
            for i, box in enumerate(self.__predicted_boxes):
                tmp.append([self.colors[i][3],box[0],box[1],box[2],box[3]])
        except:
            print("error")
        return tmp


    def __Get_NewIdentity(self):
        self.__identity += 1
        return  self.__identity
   
