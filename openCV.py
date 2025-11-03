import cv2 as cv
# img = cv.imread("D:\py\photos\VK.jpg")

def rescaleFrame(img, scale=0.55):
    width = int(img.shape[1]*scale)
    height= int(img.shape[0]*scale)
    
    dimensions = (width,height)

    return cv.resize(img,dimensions,interpolation=cv.INTER_AREA)
# resized_img= rescaleFrame(img)
# cv.imshow("vk",resized_img)
# cv.waitKey(0)

# import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame)


    cv.imshow('videos', frame)
    cv.imshow('Videos Resized', frame_resized)
    if cv.waitKey(20) & 0xFF == ord('d'):  # Press 'd' to exit
        break

capture.release()
cv.destroyAllWindows()
