# Данный код запускает определение по фотошграфии 
# (в данном случае указанную в переменной image_path)


from ultralytics import YOLO
import cv2
import cvzone
import math

model_path = 'runs/detect/train3/weights/last.pt'
image_path = 'tmp/test_photos/IMG_7657.jpg'

classNames = ["cardboard", "glass", "pet_bottle"]
img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for r in results:
    boxes = r.boxes
    drawn_y_coords = set()  # Keep track of y-coordinates already used

    for box in boxes:
        # Coords of Bounding Box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw Bounding Box
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h))

        # Confidence calc
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        # Adjust the y-coordinate to prevent overlap
        text_y = max(35, y1)

        # Check for overlap and adjust y-coordinate
        while text_y in drawn_y_coords:
            text_y += 15  # You can adjust this value based on your preference

        # Add the used y-coordinate to the set
        drawn_y_coords.add(text_y)

        # Draw text with adjusted y-coordinate
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), text_y), scale=1, thickness=1)


# Save or display the image with bounding boxes
cv2.imwrite('result.jpg', img)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
