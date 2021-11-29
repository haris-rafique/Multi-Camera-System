# -*- coding: utf-8 -*-
import numpy as np
import cv2


satellite_view=cv2.imread('C:/Users/Admin/Desktop/view.jpg')
satellite_view=cv2.resize(satellite_view,(0,0),fx=0.2,fy=0.2)

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
    

print("Press 1 for pre-recorded videos, 2 for live stream: ")
option = int(input())

if option == 2:
    source1 = cv2.VideoCapture("http://192.168.8.104:8080/video")  #the only difference between task 3 and 4 are the videos; only mapping is left
    source2 = cv2.VideoCapture("http://192.168.8.103:8080/video")
    source3=  cv2.VideoCapture(0)

elif option == 1:
    source1 = cv2.VideoCapture("C:/Users/Admin/Downloads/test1.avi")
    source2 = cv2.VideoCapture("C:/Users/Admin/Downloads/test2.avi")
    source3=source1
    

else:
    print("Invalid option entered. Exiting...")


if source1.isOpened() and source2.isOpened() and source3.isOpened():
     ret1, frame1 = source1.read()
     ret2,frame2=source2.read()
     ret3,frame3=source3.read()
else:
    ret1=False
    ret2=False
    ret3=False

img=frame2 #centre frame is taken
img=cv2.resize(img,(0,0),fx=0.9,fy=0.9)
satellite_view_cord = []
cam_view_cord = []
    
if option == 1:
    getCoordinates(img,satellite_view)
    target_points_cam1,img_points_cam1=satellite_view_cord,cam_view_cord
    H=cv2.findHomography(np.array(img_points_cam1),np.array(target_points_cam1))[0]
    np.save('Hnp',H)
    
elif option==2:
    H=np.load('C:/Users/Admin/Desktop/CVproj/Hnp.npy')
    


size1=(3*satellite_view.shape[1],satellite_view.shape[0])

size2=(satellite_view.shape[1],satellite_view.shape[0])
size3=size2
size4=size2

optputFile1 = cv2.VideoWriter(
            'Stiched1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
#optputFile2 = cv2.VideoWriter(
 #           'top1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
#optputFile3 = cv2.VideoWriter(
 #           'top2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
#optputFile3 = cv2.VideoWriter(
 #           'top3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)

while True:
    ret1, frame1 = source1.read()
    ret2,frame2=source2.read()
    ret3,frame3=source3.read()
    
    img_output1 = cv2.warpPerspective(frame1, H, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output2 = cv2.warpPerspective(frame2, H, (satellite_view.shape[1],satellite_view.shape[0]))
    img_output3 = cv2.warpPerspective(frame3, H, (satellite_view.shape[1],satellite_view.shape[0]))
    
    stiched=np.hstack([img_output1,img_output2,img_output3])   #stiching the view from all 3 cameras to get one total video output
    
    cv2.imshow("Top View", img_output1)
    cv2.imshow("Top View1",img_output2)
    cv2.imshow("Top View2",img_output3)
    cv2.imshow("Stiched",stiched)
    print(img_output1)
    optputFile1.write(stiched)
    #optputFile2.write(img_output1)
    #optputFile3.write(img_output2)
    #optputFile4.write(img_output3)
    
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break




source1.release()
source2.release()
source3.release()
optputFile1.release()
#optputFile2.release()
#optputFile3.release()
#optputFile4.release()
cv2.destroyAllWindows()
    
    

	
  
	

            
