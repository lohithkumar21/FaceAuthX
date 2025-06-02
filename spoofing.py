import math
import time
import cv2
import cvzone
from ultralytics import YOLO

# Configuration
confidence = 0.6
classNames = ["fake", "real"]
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#D-net
class DNet(nn.Module):
    def _init_(self):
        super(DNet, self)._init_()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        return self.base_model(x)
#M-net
class MNet(nn.Module):
    def _init_(self):
        super(MNet, self)._init_()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        return self.base_model(x)

#Two stage cascade
def two_stage_cascade(image, d_net, m_net, device):
    image = image.to(device)
    
    # Stage 1: D-Net for Depth-based Detection
    with torch.no_grad():
        d_net.eval()
        d_output = d_net(image)
        _, d_pred = torch.max(d_output, 1)
    
    # If D-Net detects a real face, pass to M-Net
    if d_pred.item() == 0:  # Assuming '0' is label for 'real'
        with torch.no_grad():
            m_net.eval()
            m_output = m_net(image)
            _, m_pred = torch.max(m_output, 1)
        return m_pred.item()
    
    # Otherwise, D-Net has already detected spoof
    return d_pred.item()

model = YOLO("../model/version_1.pt")

prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    d_net=DNet(img)
    m_net=MNet(img)
    result=two_stage_cascade(img,d_net=d_net,m_net=m_net)
    
    results = model(img, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            if conf > confidence:
                color = (0, 255, 0) if classNames[cls] == "real" else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%', 
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.imshow("Image", img)
    cv2.waitKey(1)
