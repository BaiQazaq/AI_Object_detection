from ultralytics import YOLO
import cv2
import cvzone
import math



cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

model = YOLO('runs/detect/train3/weights/last.pt')

classNames = ["cardboard", "glass", "pet_bottle"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Coords of Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 8, 255), 3)

            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max (0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)