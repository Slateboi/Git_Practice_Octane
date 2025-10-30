import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

video = cv.VideoCapture('13.1 test2.mp4')
# isTrue,frame = video.read()
# video.release()
# plt.imshow(frame) # (250,715) ( 1100 ,715) (600,250)
# plt.show()
def processImage(img):
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img,(5,5),3)
    canny_img = cv.Canny(blur_img,25,100)
    return canny_img

def region_of_interest(canny):
    black_img = np.zeros_like(canny)
    polygon = np.array([[ (250,715),( 1100 ,715),(600,250)]])
    mask = cv.fillPoly(black_img,polygon,(255,0,0))
    roi_img = cv.bitwise_and(canny,mask)
    return roi_img

def displayLine(frame,lines):
    black = np.zeros_like(frame)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        # print(x1,y1,x2,y2)
        cv.line(black,(x1,y1),(x2,y2),(255,0,0),10)
    return black

def optimizeLines(img,lines):
    pass


while True:
    isTrue,frame = video.read()    

    canny = processImage(frame)
    roi_img = region_of_interest(canny)
    lines = cv.HoughLinesP(roi_img,2,np.pi/180,100,np.array([]),minLineLength = 40, maxLineGap = 60) # o/p in form of  3D array[ [[x1,y1,x2,y2] , [ Line 2] ,[ Line 3]] ]
    line_detected_img = displayLine(frame,lines)

    ## Comnine 2 images 
    comnined_img = cv.addWeighted(frame,0.8,line_detected_img,1,0)

    cv.imshow("Video",comnined_img)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

