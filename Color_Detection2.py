import cv2
#import numpy as np

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg",)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

class_list = []

with open("dnn_model/classes.txt",'r') as txt:
    doc = txt.readlines()
    for items in doc:
        class_list.append(items.strip())

video = cv2.VideoCapture(0)
# video.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
while True:
    ret,frame = video.read()
    (class_id,score,bound_boxes) = model.detect(frame)
    #print(class_id)
    #print(score)
    #print(bound_boxes)
    for cl_id,scr,bbox in zip(class_id,score,bound_boxes):
        (x,y,w,h) = bbox
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(frame,str(class_list[cl_id]),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break 