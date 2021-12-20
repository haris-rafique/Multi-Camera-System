# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:29:45 2021

@author: Admin
"""

import numpy as np
import cv2

print("Press 1 for pre-recorded videos, 2 for live stream: ")
option = int(input())

if option == 2:
    source1 = cv2.VideoCapture("http://192.168.8.101:8080/video")
    source2 = cv2.VideoCapture("http://192.168.8.102:8080/video")
    source3=  source2#cv2.VideoCapture("http://192.168.8.101:8080/video")

elif option == 1:
    source1 = cv2.VideoCapture("C:/Users/Admin/Desktop/feedl11.mp4")
    source2 = cv2.VideoCapture("C:/Users/Admin/Desktop/CVproj/feedm.mp4")
    source3= cv2.VideoCapture("C:/Users/Admin/Desktop/CVproj/feedr1.mp4")
    

else:
    print("Invalid option entered. Exiting...")



if option ==1:
    satellite_view=cv2.imread('C:/Users/Admin/Desktop/plan.jpg')   #this will change with mode
    satellite_view=cv2.resize(satellite_view,(0,0),fx=0.3,fy=0.3)
    satellite_view1=cv2.imread('C:/Users/Admin/Desktop/plan.jpg')   #this will change with mode
    satellite_view1=cv2.resize(satellite_view1,(0,0),fx=0.3,fy=0.3)
    satellite_view2=cv2.imread('C:/Users/Admin/Desktop/plan.jpg')   #this will change with mode
    satellite_view2=cv2.resize(satellite_view2,(0,0),fx=0.3,fy=0.3)


elif option ==2:
    satellite_view=cv2.imread('C:/Users/Admin/Desktop/room.jpg')   #this will change with mode
    satellite_view=cv2.resize(satellite_view,(0,0),fx=0.3,fy=0.3)
    satellite_view1=cv2.imread('C:/Users/Admin/Desktop/room.jpg')   #this will change with mode
    satellite_view1=cv2.resize(satellite_view1,(0,0),fx=0.3,fy=0.3)
    satellite_view2=cv2.imread('C:/Users/Admin/Desktop/room.jpg')   #this will change with mode
    satellite_view2=cv2.resize(satellite_view2,(0,0),fx=0.3,fy=0.3)

satellite_view_cord = []
cam_view_cord = [] 
satellite_view_cord1 = []
cam_view_cord1 = []
satellite_view_cord2 = []
cam_view_cord2 = []





def satellite_on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        
        satellite_view_cord.append((x, y))
        
        cv2.circle(satellite_view, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(satellite_view, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
       
        cv2.imshow("satellite-view", satellite_view)
        print("satellite-view",x,y)
        
def cam_view_on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cam_view_cord.append((x, y))
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("cam-view", img)
        print("cam view",x,y) 
        
        
def satellite_on_EVENT_LBUTTONDOWN1(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        
        satellite_view_cord1.append((x, y))
        
        cv2.circle(satellite_view1, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(satellite_view1, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
       
        cv2.imshow("satellite-view", satellite_view1)
        print("satellite-view",x,y)
        
def cam_view_on_EVENT_LBUTTONDOWN1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cam_view_cord1.append((x, y))
        cv2.circle(img1, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img1, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("cam-view", img1)
        print("cam view",x,y) 
        
def satellite_on_EVENT_LBUTTONDOWN2(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        
        satellite_view_cord2.append((x, y))
        
        cv2.circle(satellite_view2, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(satellite_view2, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
       
        cv2.imshow("satellite-view", satellite_view2)
        print("satellite-view",x,y)
        
def cam_view_on_EVENT_LBUTTONDOWN2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cam_view_cord2.append((x, y))
        cv2.circle(img2, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img2, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("cam-view", img2)
        print("cam view",x,y) 
        
        
def getCoordinates(img,satellite_view):  
    cv2.namedWindow("satellite-view")
    cv2.setMouseCallback("satellite-view", satellite_on_EVENT_LBUTTONDOWN)
    cv2.imshow("satellite-view", satellite_view)
    cv2.namedWindow("cam-view")
    cv2.setMouseCallback("cam-view", cam_view_on_EVENT_LBUTTONDOWN)
    cv2.imshow("cam-view", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def getCoordinates1(img1,satellite_view1):  
    cv2.namedWindow("satellite-view")
    cv2.setMouseCallback("satellite-view", satellite_on_EVENT_LBUTTONDOWN1)
    cv2.imshow("satellite-view", satellite_view1)
    cv2.namedWindow("cam-view")
    cv2.setMouseCallback("cam-view", cam_view_on_EVENT_LBUTTONDOWN1)
    cv2.imshow("cam-view", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def getCoordinates2(img2,satellite_view2):  
    cv2.namedWindow("satellite-view")
    cv2.setMouseCallback("satellite-view", satellite_on_EVENT_LBUTTONDOWN2)
    cv2.imshow("satellite-view", satellite_view2)
    cv2.namedWindow("cam-view")
    cv2.setMouseCallback("cam-view", cam_view_on_EVENT_LBUTTONDOWN2)
    cv2.imshow("cam-view", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if source1.isOpened() and source2.isOpened() and source3.isOpened():
     ret1, frame1 = source1.read()
     ret2,frame2=source2.read()
     ret3,frame3=source3.read()
else:
    ret1=False
    ret2=False
    ret3=False

img=frame1 #centre frame is taken
img1=frame2
img2=frame3
img=cv2.resize(img,(0,0),fx=0.6,fy=0.6)
img1=cv2.resize(img1,(0,0),fx=0.6,fy=0.6)
img2=cv2.resize(img2,(0,0),fx=0.6,fy=0.6)

'''

'''
getCoordinates2(img2,satellite_view2)
getCoordinates1(img1,satellite_view1)
getCoordinates(img,satellite_view)
target_points_cam1,img_points_cam1=satellite_view_cord,cam_view_cord
target_points_cam2,img_points_cam2=satellite_view_cord1,cam_view_cord1
target_points_cam3,img_points_cam3=satellite_view_cord2,cam_view_cord2

H=cv2.findHomography(np.array(img_points_cam1),np.array(target_points_cam1))[0]
H1=cv2.findHomography(np.array(img_points_cam2),np.array(target_points_cam2))[0]
H2=cv2.findHomography(np.array(img_points_cam3),np.array(target_points_cam3))[0]


if option ==1:
    np.save('Hoff',H)
    np.save('H1off',H1)
    np.save('H2off',H2)
   
    
elif option ==2:
    np.save('Hon',H)
    np.save('H1on',H1)
    np.save('H2on',H2)
    H=np.load('C:/Users/Admin/Desktop/CVproj/Hon.npy')
    H1=np.load('C:/Users/Admin/Desktop/CVproj/H1on.npy')
    H2=np.load('C:/Users/Admin/Desktop/CVproj/H2on.npy')


size1=(satellite_view.shape[1],satellite_view.shape[0])



optputFile1 = cv2.VideoWriter(
            'Stiched1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)


while True:
    ret1, frame1 = source1.read()
    ret2,frame2=source2.read()
    ret3,frame3=source3.read()
    frame1=cv2.resize(frame1,(0,0),fx=0.9,fy=0.9)
    frame2=cv2.resize(frame2,(0,0),fx=0.9,fy=0.9)
    frame3=cv2.resize(frame3,(0,0),fx=0.9,fy=0.9)

    img_output1 = cv2.warpPerspective(frame1, H, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output2 = cv2.warpPerspective(frame2, H1, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output3 = cv2.warpPerspective(frame3, H2, (satellite_view.shape[1],satellite_view.shape[0]))
    
   
    
   
    
    
    avg1=cv2.addWeighted(img_output1,0.3,img_output3,0.3,0)
    summation=cv2.addWeighted(avg1,1,img_output2,0.6,0)
    rr=(np.where(img_output1>0,True,False)) & (np.where(img_output2==0,True,False)) & (np.where(img_output3==0,True,False))
    rr1=(np.where(img_output2>0,True,False)) & (np.where(img_output1==0,True,False)) & (np.where(img_output3==0,True,False))
    rr2=(np.where(img_output3>0,True,False)) & (np.where(img_output1==0,True,False)) & (np.where(img_output2==0,True,False))
    summation[rr]=img_output1[rr]
    summation[rr1]=img_output2[rr1]
    summation[rr2]=img_output3[rr2]
    
   
    
    
    
    

    
    temp=summation

    cv2.imshow("Top View", img_output1)
    cv2.imshow("Top View1",img_output2)
    cv2.imshow("Top View2",img_output3)
    temp=cv2.resize(temp,(0,0),fx=0.9,fy=0.7)
    cv2.imshow("Stiched",temp)
  
   
    optputFile1.write(summation)
   
    
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break




source1.release()
source2.release()
source3.release()
optputFile1.release()

cv2.destroyAllWindows()
