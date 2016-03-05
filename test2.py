import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


camera = PiCamera()
camera.resolution = (160, 120)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(160, 120))

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30,30),
                                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                                        )
        
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.imwrite("face", image)
    
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord("q"):
        break