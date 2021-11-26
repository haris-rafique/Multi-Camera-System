# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:29:45 2021

@author: Admin
"""

import numpy as np

import cv2
satellite_view=cv2.imread('C:/Users/Admin/Desktop/task3.jpg')

satellite_view_cord = []
cam_view_cord = [] 
def satellite_on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
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
        
        
        
def getCoordinates(img,satellite_view):  
    cv2.namedWindow("satellite-view")
    cv2.setMouseCallback("satellite-view", satellite_on_EVENT_LBUTTONDOWN)
    cv2.imshow("satellite-view", satellite_view)
    cv2.namedWindow("cam-view")
    cv2.setMouseCallback("cam-view", cam_view_on_EVENT_LBUTTONDOWN)
    cv2.imshow("cam-view", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
source1 = cv2.VideoCapture("C:/Users/Admin/Desktop/testoutput.avi")

if source1.isOpened():
     ret1, frame1 = source1.read()
else:
    ret1=False
    
    
    
    
satellite_view_cord = []
cam_view_cord = []
#img_top = cv2.resize(img_top, (700, 700)) 
img=cv2.imread('C:/Users/Admin/Desktop/1.jpg')
#img=cv2.resize(img, (700, 700)) 
getCoordinates(img,satellite_view)
target_points_cam1,img_points_cam1=satellite_view_cord,cam_view_cord


H=cv2.findHomography(np.array(img_points_cam1),np.array(target_points_cam1))[0]

#img=cv2.resize(img, (700, 400)) 


while True:
    ret1, frame1 = source1.read()
    img_output1 = cv2.warpPerspective(frame1, H, (satellite_view.shape[1],satellite_view.shape[0]))
    cv2.imshow("OUTPUT vid", img_output1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break




source1.release()
cv2.destroyAllWindows()