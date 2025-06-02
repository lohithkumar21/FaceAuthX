from time import time
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

# Configuration
classID = 0  # 0 is fake and 1 is real
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 35
debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6

# Initialize Camera and Detector
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score > confidence:
                offsetW = (offsetPercentageW / 100) * w
                x, w = int(x - offsetW), int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y, h = int(y - offsetH * 3), int(h + offsetH * 3.5)

                x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                listBlur.append(blurValue > blurThreshold)

                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', 
                                   (x, y - 0), scale=2, thickness=3)

        if save and all(listBlur):
            timeNow = str(int(time() * 1000))
            cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
            with open(f"{outputFolderPath}/{timeNow}.txt", 'a') as f:
                f.writelines(listInfo)

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)