# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:21:28 2021

@author: Admin
"""
import imutils
import numpy as np
import cv2


class Sticher:
    def __init__(self):
        self.isv3=imutils.is_cv3
        self.cachedH=None
        
    def stich(self,images,ratio=0.75,reprojThresh=4.0):
        (imageB,imageA) = images
        
        
        
        if self.cachedH is None:
            (kpsA,featuresA)=self.detectAndDescribe(imageA)
            (kpsB,featuresB)=self.detectAndDescribe(imageB)
            M=self.matchKeypoints(kpsA,kpsB,featuresA,featuresB,ratio,reprojThresh)
            if M is None:
                return None
        
            self.cachedH=M[1]
        
        result=cv2.warpPerspective(imageA,self.cachedH,(imageA.shape[1]+imageB.shape[1],imageA.shape[0]))
        result[0:imageB.shape[0],0:imageA.shape[1]]=imageB
        
        
        return result
    
    def detectAndDescribe(self,image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if self.isv3:
            descriptor= cv2.SIFT_create()
            (kps,features)= descriptor.detectAndCompute(image, None)
        else:
            detector=cv2.FeatureDetector_create("SIFT")
            kps=detector.detect(gray)
            
            extractor=cv2.DescriptorExtractor_create("SIFT")
            (kps,features)=extractor.compute(gray,kps)
            
        kps= np.float32([kp.pt for kp in kps])
        
        return (kps,features)
    
    def matchKeypoints(self,kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches=[]
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance *ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                
            if len(matches)>4:
                ptsA= np.float32([kpsA[i] for (_, i) in matches])
                ptsB = np.float32([kpsB[i] for (i, _) in matches])
                
                (H,status)=cv2.findHomography(ptsA,ptsB,cv2.RANSAC,reprojThresh)
                
                return (matches,H,status)
            
        return None
                
                
            
        
        

capture1 = cv2.VideoCapture("C:/Users/Admin/Desktop/vid2.mp4")
capture2 = cv2.VideoCapture("C:/Users/Admin/Desktop/vid1.mp4")

sticher=Sticher()
ret1,frame1 = capture1.read()
ret2,frame2 = capture2.read()

if capture1.isOpened() and capture2.isOpened():
     ret1, frame1 = capture1.read()
     ret2,frame2= capture2.read()
else:
    ret1=False
    
while True:
    ret1,frame1 = capture1.read()
    ret2,frame2 = capture2.read()
    
    frame1= cv2.resize(frame1,(0,0),fx=0.4,fy=0.4)
    frame2= cv2.resize(frame2,(0,0),fx=0.4,fy=0.4)
    images=[frame1,frame2]
    
    
    result=sticher.stich(images)
    
    if result is None:
        print("Homography cannot be computed")
        break
   
    
    cv2.imshow("Result",result)
    cv2.imshow("Left",frame1)
    cv2.imshow("Right",frame2)
    
    key =cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
   

    
    
    
cv2.destroyAllWindows()
capture1.release()
        
capture2.release()
    
    
    
    

	
  
	

            