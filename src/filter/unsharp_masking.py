import cv2 as cv

def unsharp_masking(image):
    gb = cv.GaussianBlur(image, (0, 0), 2.0)
    cv.addWeighted(image, 1.5, gb, -0.5, 0, image)
    return image
