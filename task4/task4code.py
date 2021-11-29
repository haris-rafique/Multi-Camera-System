# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 00:53:28 2021

@author: Admin
"""
import cv2
import numpy as np
from keras.models import load_model
model = load_model(r'C:/Users/Admin/Downloads/myymodel11.h5')

face_clsfr=cv2.CascadeClassifier('C:/Users/Admin/Downloads/FaceMask-Detection-master/FaceMask-Detection-master/haarcascade_frontalface_default.xml')
print("Press 1 for pre-recorded videos, 2 for live stream: ")
option = int(input())

if option == 2:
    source=cv2.VideoCapture(0)  
    source1 = source
    source2= cv2.VideoCapture("http://192.168.8.100:8080/video")

elif option == 1:
    source1 = cv2.VideoCapture("C:/Users/Admin/Desktop/CVproj/feed.mp4")
    source=source1
    source2=source1

else:
    print("Invalid option entered. Exiting...")


satellite_view=cv2.imread('C:/Users/Admin/Desktop/view.jpg')
satellite_view=cv2.resize(satellite_view,(0,0),fx=0.2,fy=0.2)


size1=(3*satellite_view.shape[1],satellite_view.shape[0])
optputFile1 = cv2.VideoWriter(
            'stitchedoutput.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
        

labels_dict={1:'MASK',0:'NO MASK'}
H=np.load('C:/Users/Admin/Desktop/CVproj/Hnp.npy')
color_dict={1:(0,255,0),0:(0,0,255)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.23,4)   
    ret1,img1=source1.read()
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    faces1=face_clsfr.detectMultiScale(gray1,1.23,4) #best til now
    ret2,img2=source2.read()
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    faces2=face_clsfr.detectMultiScale(gray2,1.23,4) #best til now
    
    FaceArray=[faces,faces1,faces2]
    GrayArray=[gray,gray1,gray2]
    imgArray=[img,img1,img2]
  
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
            midpoint=((2*x+w)//2,(2*y+h)//2)
            print(midpoint)
            
            cv2.circle(imgArray[i],midpoint , 5, color_dict[label], thickness=-1)
           
            cv2.putText(imgArray[i], labels_dict[label], (midpoint[0], midpoint[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    
    
    img_output1 = cv2.warpPerspective(imgArray[0], H, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output2 = cv2.warpPerspective(imgArray[1], H, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output3 = cv2.warpPerspective(imgArray[2], H, (satellite_view.shape[1],satellite_view.shape[0]))
            
    stiched=np.hstack([img_output1,img_output2,img_output3])
    
    cv2.imshow('Feed1',img_output1)
    cv2.imshow('Feed2',img_output2)
    cv2.imshow('Feed3',img_output3)
    cv2.imshow('Stiched',stiched)
    optputFile1.write(stiched)
   
    
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
optputFile1.release()
source1.release()
source2.release()
