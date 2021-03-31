import cv2

class BackgroundRemoval:
    def __init__(self, background):
        self.background = background
        self.grayBackground = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    def background_removal(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(self.grayBackground, grayImage)
        ret,thresh1 = cv2.threshold(sub,10,255,cv2.THRESH_BINARY)
        res = cv2.bitwise_and(image, image, mask=thresh1)
        return res
