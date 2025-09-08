#!/opt/anaconda3/bin/python3

import cv2
import numpy as np
import json
import os
import sys

def get_object_cascades(filename: str) -> dict:
    object_cascades = {}
    with open(filename, 'r') as fs:
        object_cascades = json.load(fs)
    if len(object_cascades.keys()) > 0:
        for object_cascade_name in object_cascades.keys():
            # Path to the Haar cascade file for object detection
            cascaderootpath="/Users/theodorehuppert/Desktop/ECE1390/LectureNotebooks/haar_xml/"
            object_cascade_path = cascaderootpath + "/" + object_cascades[object_cascade_name]
            print(object_cascade_path)
            if os.path.exists(object_cascade_path):
                # Create a object cascade classifier
                object_cascades[object_cascade_name] = cv2.CascadeClassifier(object_cascade_path)
    else:
        raise ValueError('Load cascades into cascades.json.')
    return object_cascades

if __name__ == "__main__":
    
    # Font for displaying text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Initialize object cascade classifiers with relative objects names
    cascadefile="/Users/theodorehuppert/Desktop/ECE1390/LectureNotebooks/Lectures/Lecture16/cascades.json"
    object_cascades = get_object_cascades(cascadefile)

    # Video Capture from the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height
    
    # Minimum width and height for the window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        # Read a frame from the camera
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        for object_cascade_name in object_cascades.keys():
            object_cascade = object_cascades[object_cascade_name]

            # Detect objects in the frame
            objects = object_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )
        
            for (x, y, w, h) in objects:

                # Draw a rectangle around the detected object
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
                # Display the object name on the image
                cv2.putText(img, object_cascade_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    
        # Display the image with rectangles around faces
        cv2.imshow('camera', img)
    
        # Press Escape to exit the webcam / program
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    
    print("\n [INFO] Exiting Program.")
    # Release the camera
    cam.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
