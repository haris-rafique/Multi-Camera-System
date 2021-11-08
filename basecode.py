import cv2


def main():

    print("Press 1 for pre-recorded videos, 2 for live stream: ")
    option = int(input())

    if option == 1:
        # Record video
        windowName = "Sample Feed from Camera 1"
        windowName1 = "Sample Feed from Camera 2"
        windowName2 = "Sample Feed from Camera 3"
        cv2.namedWindow(windowName)
        cv2.namedWindow(windowName1)
        cv2.namedWindow(windowName2)

        capture1 = cv2.VideoCapture("C:/Users/Admin/Downloads/Stream1Recording.avi")  
        capture2 = cv2.VideoCapture("C:/Users/Admin/Downloads/Stream2Recording.avi")   
        capture3 = cv2.VideoCapture("C:/Users/Admin/Downloads/Stream3Recording.avi")
       
        

        # check if feed exists or not for camera 1, 2 and 3
        if capture1.isOpened() and capture2.isOpened() and capture3.isOpened():
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
        else:
            ret1 = False
            ret2=False
            ret3=False
        while True:
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            
            frame1=cv2.resize(frame1,(500,500))
            frame2=cv2.resize(frame2,(500,500))
            frame3=cv2.resize(frame3,(500,500))
            # sample feed display from camera 1, 2 and 3 simultaneously
            cv2.imshow(windowName, frame1)
            cv2.imshow(windowName1, frame2)
            cv2.imshow(windowName2, frame3)
          
            

            # q key to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        

        capture1.release()
        
        capture2.release()
        
        capture3.release()
       
        cv2.destroyAllWindows()

    elif option == 2:
        # live stream
        windowName1 = "Live Stream Camera 1"
        windowName2 = "Live Stream Camera 2"
        windowName3 = "Live Stream Camera 3"
        cv2.namedWindow(windowName1)
        cv2.namedWindow(windowName2)
        cv2.namedWindow(windowName3)

        capture1 = cv2.VideoCapture("http://192.168.131.216:8080/video")  # IP cameras IP addresses
        capture2 = cv2.VideoCapture("http://192.168.131.100:8080/video")
        capture3 = cv2.VideoCapture("http://192.168.10.17:8080/video")
        
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)
        
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)
        
        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)
        
        
        optputFile1 = cv2.VideoWriter(
            'Stream1Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
        
        optputFile2 = cv2.VideoWriter(
            'Stream2Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)
        
        optputFile3 = cv2.VideoWriter(
            'Stream3Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size3)
        
       
        
        if capture2.isOpened() and capture1.isOpened() and capture3.isOpened():  # check if feed exists or not for camera 1
            ret2, frame2 = capture2.read()
            ret1,frame1 = capture1.read()
            ret3,frame3 = capture3.read()
        else:
            ret2 = False
            ret1=  False
            ret3=   False

        while True:  # and ret2 and ret3:
            ret2, frame2 = capture2.read()
            ret1,frame1 = capture1.read()
            ret3,frame3 = capture3.read()
            frame1=cv2.resize(frame1,(500,500))
            frame2=cv2.resize(frame2,(500,500))
            frame3=cv2.resize(frame3,(500,500))
            cv2.imshow(windowName2, frame2)
            
            cv2.imshow(windowName1, frame1)
            cv2.imshow(windowName2, frame3)
            optputFile1.write(frame1)
            optputFile2.write(frame2)
            optputFile3.write(frame3)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                
                break
       # frame of size is being created and stored in .avi file
        

        capture1.release()
        optputFile1.release()
        capture2.release()
        optputFile2.release()
        capture3.release()
        optputFile3.release()
        
        cv2.destroyAllWindows()

    else:
        print("Invalid option entered. Exiting...")


main()
