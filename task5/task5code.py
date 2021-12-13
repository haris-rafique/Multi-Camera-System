# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:47:28 2021

@author: Admin
"""

MODEL_PATH = "C:/Users/Admin/Desktop/social-distance-detector-main/yolo-coco"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = True

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 70

from scipy.spatial import distance as dist
import numpy as np
import cv2
import os

def detect_people(frame, net, ln, personIdx=0):
	# grab the dimensions of the frame and  initialize the list of
	# results
	(H, W) = frame.shape[:2]
	results = []

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
	boxes = []
	centroids = []
	confidences = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
			if classID == personIdx and confidence > MIN_CONF:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results


labelsPath= os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    
    
ln = net.getLayerNames()
ln = [ln[[i][0] - 1] for i in net.getUnconnectedOutLayers()]
print("Press 1 for pre-recorded videos, 2 for live stream: ")
option = int(input())

if option == 2:
    vs=cv2.VideoCapture(0)  
    vs1 = vs
    vs2= cv2.VideoCapture("http://192.168.8.100:8080/video")

elif option == 1:
    vs = cv2.VideoCapture("C:/Users/Admin/Desktop/CVproj/feed.mp4")
    vs1=vs
    vs2=vs

else:
    print("Invalid option entered. Exiting...")



writer=None
writer1=None
writer2=None
while True:
    (grabbed, frame) = vs.read()
    (grabbed1,frame1) = vs1.read()
    (grabbed2,frame2) = vs2.read()
    
    framearray=[frame,frame1,frame2]
    if not grabbed or not grabbed1 or not grabbed2:
        break
    
    for k in range(len(framearray)):
        framearray[k]=cv2.resize(framearray[k],(0,0),fx=0.4,fy=0.4)
    
        results = detect_people(framearray[k], net, ln, personIdx=LABELS.index("person"))
    
        violate=set()
        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
        
            for i in range(0, D.shape[0]):
                for j in range(i+1,D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color=(0,255,0)
        
            if i in violate:
                color=(0,0,255)
            
            cv2.rectangle(framearray[k], (startX, startY), (endX, endY), color, 2)
            cv2.circle(framearray[k], (cX, cY), 5, color, 1)
        
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(framearray[k], text, (10, framearray[k].shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    
    
        cv2.imshow("Frame{}".format(k+1), framearray[k])
    if writer is None:
            
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('detection1.avi', fourcc, 15,(framearray[0].shape[1], framearray[0].shape[0]), True)
        writer1 = cv2.VideoWriter('detection2.avi', fourcc, 15,(framearray[1].shape[1], framearray[1].shape[0]), True)
        writer2 = cv2.VideoWriter('detection3.avi', fourcc, 15,(framearray[2].shape[1], framearray[2].shape[0]), True)
        
    if writer is not None:
        writer.write(framearray[0])
        writer1.write(framearray[1])
        writer2.write(framearray[2])
    
    
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
            

