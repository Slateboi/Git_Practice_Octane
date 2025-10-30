import cv2 as cv
import numpy as np
def canny(image):
    gray = cv.cvtColor(lane_image, cv.COLOR_RGB2GRAY)#grayscale
    blur= cv.GaussianBlur(gray, (13,13),0)#blur
    canny = cv.Canny(blur,50,150)#canny
    return canny



def rescaleFrame(image, scale=0.40):
            width = int(image.shape[1]*scale)
            height= int(image.shape[0]*scale)
            dimensions = (width,height)
            return cv.resize(image,dimensions,interpolation=cv.INTER_AREA)

def region_of_interest(image):
        height = image.shape[0]
        polygon= np.array([[(600,height),(1600,height),(800,300)]])
        mask =np.zeros_like(image)
        cv.fillPoly(mask,polygon,255)
        return mask
        

image = cv.imread("D:\py\photos\images (2).jpg")
lane_image= np.copy(image)
gray = cv.cvtColor(lane_image, cv.COLOR_RGB2GRAY)#grayscale
blur= cv.GaussianBlur(gray, (13,13),0)#blur
canny = canny(lane_image)
z= region_of_interest(canny)

resized_image = rescaleFrame(z)
cv.imshow('result',resized_image)
cv.waitKey(0)