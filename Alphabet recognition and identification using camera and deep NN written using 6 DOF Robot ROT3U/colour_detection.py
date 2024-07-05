import cv2
import numpy as np
import os

# Set frame dimensions
frameWidth = 640
frameHeight = 480

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# Blue color range in HSV
blueLower = np.array([100, 150, 0])
blueUpper = np.array([140, 255, 255])
blueColorValue = [255, 0, 0]  # BGR for blue

# List to store detected points
detectedPoints = []  # [x, y]

# Directory to save captured images
# output_dir = "captured_images"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)


def findBlueColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, blueLower, blueUpper)
    x, y = getContours(mask)
    if x != 0 and y != 0:
        return [x, y], mask
    return [], mask


def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y


def drawOnCanvas(img, points, colorValue):
    for point in points:
        cv2.circle(img, (point[0], point[1]), 10, colorValue, cv2.FILLED)


while True:
    success, img = cap.read()
    if not success:
        break

    # Mirror the image
    img = cv2.flip(img, 1)
    imgResult = img.copy()
    newPoint, mask = findBlueColor(img)
    if newPoint:
        detectedPoints.append(newPoint)
    if detectedPoints:
        drawOnCanvas(imgResult, detectedPoints, blueColorValue)

    cv2.imshow("Result", imgResult)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        # Create an image with only the blue parts
        blueOnly = cv2.bitwise_and(img, img, mask=mask)
        # Create a black background
        # blackBackground = np.zeros_like(img)
        # Create a white background
        whiteBackground = np.ones_like(img) * 255
        # Combine blue parts with the black background
        blueOnWhite = cv2.bitwise_or(blueOnly, whiteBackground)

        # Draw detected points on the captured image
        drawOnCanvas(blueOnWhite, detectedPoints, blueColorValue)

        # Save the image
        # img_name = os.path.join(output_dir, f"blue_detected_{cv2.getTickCount()}.png")
        img_name = "blue_detected_"+str(cv2.getTickCount())+".png"
        cv2.imwrite(img_name, blueOnWhite)
        print(f"Image saved: {img_name}")

        # Display the mask and the overlayed image for debugging
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Blue on Black", blueOnBlack)
        cv2.waitKey(0)  # Wait until a key is pressed to continue

cap.release()
cv2.destroyAllWindows()
