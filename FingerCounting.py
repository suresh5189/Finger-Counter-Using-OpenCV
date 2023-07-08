import cv2
import time
import os
import HandTrackingModule as htm

widthCamera, heightCamera = 1920, 1080

cap = cv2.VideoCapture(0)
cap.set(3, widthCamera)
cap.set(4, heightCamera)

folderPath = "FingerImage"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

previousTime = 0

detector = htm.handDetection(detectionCon=0.75)

for imgPath in myList:
    image = cv2.imread(folderPath + "/" + imgPath)
    # print(folderPath + "/" + imgPath)
    overlayList.append(image)

# print(len(overlayList))

tipIds = [4,8,12,16,20]

while True:
    success,img = cap.read()
    img = detector.findhands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList)
    
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        
        # 4 Fingers 
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        height,width, channel = overlayList[totalFingers-1].shape
        img[0:height,0:width] = overlayList[totalFingers-1]
        
        cv2.rectangle(img,(20,460),(175,630),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,600),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),20)
    
    currentTime = time.time()
    FPS = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img,f'FPS:{int(FPS)}',(310,40),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    
    cv2.imshow("Output",img)
    cv2.waitKey(1)