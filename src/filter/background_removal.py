import cv2
import numpy as np


class BackgroundRemoval:
    def __init__(self, background):
        self.background = background
        self.grayBackground = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    def get_mask(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(self.grayBackground, grayImage)
        ret, mask = cv2.threshold(sub, 10, 255, cv2.THRESH_BINARY)
        cv2.erode(mask, np.ones((3, 3), np.uint8), mask)
        cv2.dilate(mask, np.ones((3, 3), np.uint8), mask)
        return grayImage

    def background_removal(self, image):
        mask = self.get_mask(image)
        res = cv2.bitwise_and(image, image, mask=mask)
        return res
