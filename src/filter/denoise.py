import cv2 as cv

def unsharp_masking(image):
    dst = cv2.fastNlMeansDenoising(image,None,10,10,7)
    return dst
