import cv2
import numpy as np
import matplotlib.pyplot as plt

drawing = False
cv2.namedWindow('Image')

x1, x2, y1, y2 = None, 0, 0, 0

def drawLine(event, x, y, flags, param):
    global x2, y2, drawing
    
    if event == cv2.EVENT_MOUSEMOVE:
        x2, y2 = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


imageName = 'Papa.jpg';
originalImage = cv2.imread(imageName, 0);

# image = np.zeros(originalImage.shape, np.uint8) # Black Image
imageMask = np.zeros(originalImage.shape, np.uint8)

cv2.setMouseCallback('Image', drawLine)

while True:

    if drawing:
        if x1 is not None:
            originalImage = cv2.line(originalImage, (x1,y1), (x2,y2), [255,255,255], 10)
            imageMask = cv2.line(imageMask, (x1,y1), (x2,y2), [255,255,255], 10)
            
            x1 = x2
            y1 = y2
        else:
            x1, y1 = x2, y2
    else:
        x1 = None
    
    cv2.imshow('Image', originalImage)
    
    k = cv2.waitKey(1)
    
    if k == ord('c'):
        originalImage = cv2.imread(imageName)
        imageMask = np.zeros(originalImage.shape, np.uint8)
    elif k == ord('s'):
        cv2.imwrite("mask" + imageName, imageMask)
        break
    elif k == 27:
        break

cv2.destroyAllWindows()