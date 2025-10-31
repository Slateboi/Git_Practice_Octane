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
    canny_roi_img = cv.bitwise_and(canny,mask)
    return canny_roi_img

def displayLine(frame,lines):
    black = np.zeros_like(frame)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        # print(x1,y1,x2,y2)
        cv.line(black,(x1,y1),(x2,y2),(255,0,0),10)
    return black

def optimizeLines(img,lines):

    # plt.imshow(img)
    # plt.show() # line on the left will have -ve slope & line on right will have +ve slope

    left_slope_list = []
    left_intercept_list = []

    right_slope_list = []
    right_intercept_list = []

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4) #
        line_parameter = np.polyfit((x1,x2),(y1,y2),1) # 1 represnts the degree of poly

        print(line_parameter) ## o/p [slope,intercept]

        if line_parameter[0] <0 :
            left_slope_list.append(line_parameter[0])
            left_intercept_list.append(line_parameter[1])
        else:
            right_slope_list.append(line_parameter[0])
            right_intercept_list.append(line_parameter[1])

    avg_left_slope = sum(left_slope_list)/len(left_slope_list)
    avg_left_intercept = sum(left_intercept_list)/len(left_intercept_list)

    avg_right_slope = sum(right_slope_list)/len(right_slope_list)
    avg_right_intercept = sum(right_intercept_list)/len(right_intercept_list)

    print(avg_left_slope,avg_left_intercept)

    ## finding points for left line
    x1 = 300
    x2 = 500
    y1 = avg_left_slope*x1 + avg_left_intercept
    y2 = avg_left_slope*x2 + avg_left_intercept

    ## finding points for right line
    x3 = 600
    x4 = 1000
    y3 = avg_right_slope*x3 + avg_right_intercept
    y4 = avg_right_slope*x4 + avg_right_intercept

    left_line_img = cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
    optimized_line_img = cv.line(left_line_img,(int(x3),int(y3)),(int(x4),int(y4)),(255,0,0),10)

    return optimized_line_img
    

while True:
    isTrue,frame = video.read()    
    frame2 = np.copy(frame)
    canny = processImage(frame)
    canny_roi_img = region_of_interest(canny)
    lines = cv.HoughLinesP(canny_roi_img,2,np.pi/180,100,np.array([]),minLineLength = 20, maxLineGap = 60) # o/p in form of  3D array[ [[x1,y1,x2,y2] , [ Line 2] ,[ Line 3]] ]
    line_detected_img = displayLine(frame,lines)

    optimized_line_img = optimizeLines(frame,lines)

    ## Comnine 2 images 
    comnined_img = cv.addWeighted(frame,0.8,line_detected_img,1,0)
    cv.imshow("Lane Detection",comnined_img)

    comnined__optimized_img = cv.addWeighted(frame2,0.8,optimized_line_img,1,0)
    cv.imshow("Optimized Lane Detection",comnined__optimized_img)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break

