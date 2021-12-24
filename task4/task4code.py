# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 00:53:28 2021

@author: Admin
"""
import cv2
import numpy as np
from keras.models import load_model
model = load_model(r'C:/Users/Admin/Desktop/myymodel.h5')

face_clsfr=cv2.CascadeClassifier('C:/Users/Admin/Downloads/FaceMask-Detection-master/haarcascade_frontalface_default.xml')
print("Press 1 for pre-recorded videos, 2 for live stream: ")
option = int(input())

if option == 2:
    source=cv2.VideoCapture("http://192.168.8.103:8080/video")
    source1 = cv2.VideoCapture(0)
    source2= source
    H=np.load('C:/Users/Admin/Desktop/CVproj/Hon.npy')
    H1=np.load('C:/Users/Admin/Desktop/CVproj/H1on.npy')
    H2=np.load('C:/Users/Admin/Desktop/CVproj/H2on.npy')

elif option == 1:
    
    source=cv2.VideoCapture("C:/Users/Admin/Desktop/feedl11.mp4")  #left video
    source1 = cv2.VideoCapture("C:/Users/Admin/Desktop/CVproj/feed.mp4") #middle video
    source2=cv2.VideoCapture("C:/Users/Admin/Desktop/CVproj/feedr1.mp4") #right video
    H=np.load('C:/Users/Admin/Desktop/Hoff.npy')
    H1=np.load('C:/Users/Admin/Desktop/H1off.npy')
    H2=np.load('C:/Users/Admin/Desktop/H2off.npy')
else:
    print("Invalid option entered. Exiting...")


satellite_view=cv2.imread('C:/Users/Admin/Desktop/plan.jpg')
satellite_view=cv2.resize(satellite_view,(0,0),fx=0.3,fy=0.3)


size1=(satellite_view.shape[1],satellite_view.shape[0])
optputFile1 = cv2.VideoWriter(
            'stitchedoutput.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
        

labels_dict={1:'MASK',0:'NO MASK'}
#H=np.load('C:/Users/Admin/Desktop/CVproj/Hnp.npy')
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
            
            cv2.circle(imgArray[i],midpoint , 10, color_dict[label], thickness=-1)
           
            cv2.putText(imgArray[i], labels_dict[label], (midpoint[0], midpoint[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color_dict[label],2)
        
    
    
    img_output1 = cv2.warpPerspective(imgArray[0], H, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output2 = cv2.warpPerspective(imgArray[1], H1, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output3 = cv2.warpPerspective(imgArray[2], H2, (satellite_view.shape[1],satellite_view.shape[0]))
            
    #stiched=np.hstack([img_output1,img_output2,img_output3])
    avg1=cv2.addWeighted(img_output1,0.3,img_output3,0.3,0)
    summation=cv2.addWeighted(avg1,1,img_output2,0.5,0)
    r=(np.where(img_output1>0,True,False)) & (np.where(img_output2==0,True,False)) & (np.where(img_output3==0,True,False))
    r1=(np.where(img_output2>0,True,False)) & (np.where(img_output1==0,True,False)) & (np.where(img_output3==0,True,False))
    r2=(np.where(img_output3>0,True,False)) & (np.where(img_output1==0,True,False)) & (np.where(img_output2==0,True,False))
    summation[r]=img_output1[r]
    summation[r1]=img_output2[r1]
    summation[r2]=img_output3[r2]
    
    
    
    
    temp=summation
    cv2.imshow('Feed1',img_output1)
    cv2.imshow('Feed2',img_output2)
    cv2.imshow('Feed3',img_output3)
   
    temp=cv2.resize(temp,(0,0),fx=0.9,fy=0.7)
    cv2.imshow("Stiched",temp)
    optputFile1.write(summation)
   
    
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
optputFile1.release()
source1.release()
source2.release()
