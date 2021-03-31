import cv2 as cv

def erode(image):
    dt = cv.distanceTransform(image, cv.DIST_L2, 5)
    # TODO: DO
