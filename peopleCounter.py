#import the necessary packages
from cv2 import non_max_suppression
import numpy as np
import imutils #lets you perorm transfomations from the results
import cv2 #OpenCV Python wrapper
import requests #lets yousend data/results
import time
import argparse #lets you read commands from the command terminal inside the script


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


DEVICE = "detector" #Device where the result will be stored
VARIABLE = "people" #Variable where the result will be stored

#Opencv pre-traied SVM with Histogram Oriented Object detector (HOG for short) people features
#OpenCV has an efficient way to combine the HOG algorithm with a support vector machine (SVM), a machine learning technique
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detector(image):
    #load the image and resize it to reduce detection time and improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    clone = image.copy()
    #detect people in the image
    (rects,weights) = HOGCV.detecMultiScale(image, winStride = (8,8)) #detectMultiScale() method is from the HOG object and lets me analyze the image and know if a person exists using the classification from the SVM
    #padding = (32,32, scale = 1.05)

#applies non-max suppression from imutils package to kick off overlapped boxes
#uses a large overlap threshold to maintain overlapping bodes that are still people
    rects = np.array([[x,y,x + w,y+h] for (x,y,w,h) in rects])
    result = non_max_suppression(rects,probs=None, overlapThresh = 0.65)
    return result

#define a function to read an image from a local file and detect any people in it
#calling the detector() function and adding a loop to paint the round-boxes for the detector
#returns the number of detected boxes and the image with the painted detection
def localDetect(image_path):
    result = []
    image = cv2.imread(image_path)
    if len(image) <= 0:
        print ("[ERROR could not read your local image")
        return result
    print("[INFO] Detecting people")
    result = detector(image)

    #shows the result
    for (xA, yA, xB, yB) in result: cv2.rectangle(image, (xA, yA), (xB,yB), (0,255,0),2)
    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (result, image)

#calling the detector() method and paint boxes around the detectPeople, #convert_to_base64() converts the image to a base 64 string
def cameraDetect(token,device,variable, sample_time=5):

    cap = cv2.VideoCapture(0)
    init = time.time()

    #Allowed sample time for Ubidots is 1 dot/second
    if sample_time < 1:
        sample_time = 1

    while(True):
        # Capture frame-byframe
        ret, frame = cap.read()
        frame = imutils.resize(frame,width=min(400,frame.shape[1]))
        result = detector(frame.copy())

        #shows the result
        for (xA, yA, xB, yB) in result:
            cv2.rectangle(frame, (xA, yA), (xB,yB), (0,255,0),2)
        cv2.imshow('frame',frame)

        #sends results
        if time.time() - init >= sample_time:
            print("[INFO] Sending actual frame results")
            b64 = convert_to_base64(frame)
            context = {"image":b64}
            sendToUbidots(token,device,variable, len(result), context=context)
            init = time.time()
        if cv2.waitKey(1) & OxFF == ord('q'):
            break

    #when everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def convert_to_base64(image):
    image =imutils.resize(image, width=400)
    img_str = cv2.imencode('.png',image)[1].tostring()
    b64=base64.b64encode(img_str)
    return b64.decode('utf-8')

#gets the arguments inserted through terminal to trigger a routine that searches for people in a locally stored image file
def detectPeople(args):
    image_path = args["image"]

    # Routine to read local image
    if image_path != None:
        print("[INFO] Image path provided, attempting to read image")
        (result, image) = localDetect(image_path)
        print("[INFO] sending results")
        # Converts the image to base 64 and adds it to the context
        b64 = convert_to_base64(image)
        context = {"image": b64}

def main():
    args = argsParser()
    detectPeople(args)

#main() function calls the arguments from the console and launches the specified routine
if __name__ == '__main__':
    main()

# show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))
    # show the output images
    cv2.imshow("Before", clone)
    cv2.imshow("After", image)
    cv2.waitKey(0)

    #return the number of people:
