import cv2
import time
vidcap = cv2.VideoCapture('/home/mgharasu/Videos/Wo4.avi')
prev = 0
counter = 0
sec = 0
frameRate = 1/50#it will capture image in each 0.5 second
frame_counter = 0

def getFrame(sec):
    global frame_counter

    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        #cv2.imwrite("frame "+str(sec)+" sec.jpg", image)     # save frame as JPG file        
        frame_counter += 1
        image = cv2.putText(image,str(frame_counter),(60,60),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),1)
        cv2.imshow("output", image)
        
        if cv2.waitKey(1) & 0xFF == ord('a'):
            global prev, counter
            tc=time.time()
            print("rep No:", str(counter),"time: ",str(tc-prev))
            prev = tc
            counter += 1

    return hasFrames


success = getFrame(sec)

while success:
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
