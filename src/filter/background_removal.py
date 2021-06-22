import cv2
import numpy as np


class BackgroundRemoval:
    def __init__(self, background):
        self.background = background
        self.grayBackground = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    def get_mask(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(self.grayBackground, grayImage)
        ret, thresh1 = cv2.threshold(sub, 10, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh1, np.ones((3,3), np.uint8), thresh1)
        cv2.dilate(thresh1, np.ones((3,3), np.uint8), thresh1)
        return thresh1

    def background_removal(self, image):
        res = cv2.bitwise_and(image, image, mask=self.get_mask(image))
        return res
