# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 00:53:28 2021

@author: Admin
"""
import cv2
import numpy as np
from keras.models import load_model
model = load_model(r'C:/Users/Admin/Downloads/mymodel.h5')

face_clsfr=cv2.CascadeClassifier('C:/Users/Admin/Downloads/haarcascade_frontalface_default.xml')
print("Press 1 for pre-recorded videos, 2 for live stream: ")
option = int(input())

if option == 2:  
    source=cv2.VideoCapture("http://192.168.8.105:8080/video")  #online mode
    source1 = cv2.VideoCapture("http://192.168.7.105:8080/video")
    source2 = cv2.VideoCapture("http://192.168.9.105:8080/video")

elif option == 1:  #offline mode
    source1 = cv2.VideoCapture("C:/Users/Admin/Desktop/livefeed.mp4")  #tested on prerecorded video! 
    source= cv2.VideoCapture("C:/Users/Admin/Desktop/test1.avi")
    source2=cv2.VideoCapture("C:/Users/Admin/Desktop/test2.avi")


width1 = int(source.get(3))
height1 = int(source.get(4))
size1 = (width1, height1)
        
width2 = int(source1.get(3))
height2 = int(source1.get(4))       
size2 = (width2, height2)

width3 = int(source2.get(3))
height3 = int(source2.get(4))       
size3 = (width3, height3)


optputFile1 = cv2.VideoWriter(
            'test1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
        
optputFile2 = cv2.VideoWriter(
            'test2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)

optputFile3 = cv2.VideoWriter(
            'test3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)

labels_dict={1:'MASK',0:'NO MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.23,4)
    ret1,img1=source1.read()
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    faces1=face_clsfr.detectMultiScale(gray1,1.23,4) 
    
    
    
    FaceArray=[faces,faces1]
    GrayArray=[gray,gray1]
    imgArray=[img,img1]
  
    for i in range(0,len(FaceArray)):
        

        for (x,y,w,h) in FaceArray[i]:  
    
            face_img=GrayArray[i]
            face_img=face_img[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)
            
            if np.round(result)==1:
                label=1
            elif np.round(result)==0:
                label=0
      
            cv2.rectangle(imgArray[i],(x,y),(x+w,y+h),color_dict[label],2)
            cv2.putText(imgArray[i], labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    
        
    
    cv2.imshow('Feed1',imgArray[0])
    cv2.imshow('Feed2',imgArray[1])
    cv2.imshow('Feed3',imgArray[2])
    optputFile1.write(imgArray[0])
    optputFile2.write(imgArray[1])
    optputFile3.write(imgArray[2])
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
optputFile1.release()
source1.release()
optputFile2.release()
source2.release()
optputFile3.release()
