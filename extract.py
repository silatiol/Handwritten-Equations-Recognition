import cv2
import numpy as np

image2 = cv2.imread("./img/2011/training/920.png")
image2 = cv2.bitwise_not(image2)
image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

cv2.imshow("test", image)

elements, hierachy = cv2.findContours(
    image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

kernel = np.ones((3, 3), np.uint8)

print(hierachy)

i = 0
for e in elements:

    x, y, w, h = cv2.boundingRect(e)

    # k = cv2.drawContours(cv2.bitwise_not(
    #    np.zeros_like(image)), [e], 0, (0, 255, 0), 2)
    k = cv2.fillPoly(cv2.bitwise_not(
        np.zeros_like(image)), [e], color=(0, 255, 0))
    k = cv2.bitwise_not(k)
    #k = cv2.morphologyEx(k, cv2.MORPH_OPEN, kernel)
    #k = cv2.morphologyEx(k, cv2.MORPH_CLOSE, kernel)
    k = cv2.erode(k, np.ones((1, 1), np.uint8), iterations=1)

    if(w > h):
        while(w/h > 2):
            w = int(w*0.9)
        scale_percent = 28/w
    else:
        scale_percent = 28/h

    k = k[y:y+h, x:x+w]
    width = int(k.shape[1] * scale_percent)
    height = int(k.shape[0] * scale_percent)

    k = cv2.resize(k, (width, height))

    final = np.zeros((28, 28))
    y1, x1 = int((final.shape[0]-k.shape[0]) /
                 2), int((final.shape[1]-k.shape[1])/2)
    final[y1:y1+k.shape[0], x1:x1+k.shape[1]] = k
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    #k = cv2.resize(k, (28, 28))
    #k = cv2.erode(k, kernel, iterations=1)
    cv2.imshow("test"+str(i), final)

    img2 = cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 1)

    i += 1

cv2.imshow("result", image2)

while(1):
    q = cv2.waitKey(1) & 0xFF
    if q == 27:
        break
