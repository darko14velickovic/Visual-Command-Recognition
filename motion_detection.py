import argparse
import datetime


import imutils
import time
import cv2
import numpy as np

from image_processor import Trainer

import skvideo.io



# printing util
def print_sign_top_text(frame, showing_what):

    background = frame[0: 50, 0: frame.shape[1]]
    # cv2.imshow("Testing", background)
    avg = np.average(background)
    if avg < 125:
        avg = 0
    else:
        avg = 255
    cv2.putText(frame, "Showing: {}".format(showing_what), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255 - avg, 255 - avg, 255 - avg), 2)

# printing util
def print_time_bottom(frame):
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# runs evaluation on a given part of the image
def check_roi(roi):
    roi = np.divide(roi, 1000.)
    prediction = neural_net.evaluate_img(roi)
    # print (prediction[0])

    max_index = np.argmax(prediction)

    certainty = prediction[0][max_index]
    class_string = ""
    if max_index == 3:
        class_string = "OTHER"
    elif max_index == 0:
        # print_sign_top_text(frame, "FIST: " + certainty)
        class_string = "FIST"
    elif max_index == 1:
        # print_sign_top_text(frame, "PALM: " + certainty)
        class_string = "PALM"
    elif max_index == 2:
        # print_sign_top_text(frame, "POINT: " + certainty)
        class_string = "POINT"

    print("Predicting class: %s with certainty :%f", class_string, certainty)
    return class_string, certainty

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--minarea", type=int, default=60, help="minimum area size")
ap.add_argument("-m", "--network", type=str, default="hands", help="name of the network model in model folder")
ap.add_argument("-d", "--dimension", type=int, default=60, help="dimension of the network output layer")

args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = skvideo.io.VideoCapture(0)
    # camera.set(4 , 352)
    # camera.set(5 , 288)
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    camera = skvideo.io.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None

print("Loading neural net: " + args.get("network"))
# load convolution neural net from model folder
neural_net = Trainer.CnnTrainer(args.get("minarea"), args.get("dimension"), args.get("network"), False)
neural_net.load_cnn(args.get("network"))
print("Neural net loaded")


show_string = ""
percent = ""

min_x = 999999999
min_y = 999999999

print("Min area argument: ")
print(args["minarea"])

# loop over the frames of the video
while True:
    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print("Cant grab from camera!")
        break

    # resize the frame, convert it to grayscale, and blur it
    # print("Got image")
    frame = imutils.resize(frame, width=800)
    # print("imutils resize done")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 55, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (im2, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    # save frame for the next comparison
    firstFrame = gray
    dim = args.get("minarea")

    im_x = 999999
    im_y = 999999

    # loop over the contours
    for c in cnts:

        # a lot of changes, break the loop
        if cnts.__len__() > 3:
            break

        # check size of contours, if the size is to big or too small, continue
        if cv2.contourArea(c) < args["minarea"] or cv2.contourArea(c) > args["minarea"] * 3:
            # print_time_bottom(frame)
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)

        # adjust the coordinates of the center point
        if x < im_x:
            min_x = x
        if y < im_y:
            min_y = y

    labels = []
    percs = []

    cv2.rectangle(frame, (min_x, min_y), (min_x + dim, min_y + dim), (0, 255, 0), 2)
    # roi = frame[min_x: min_x + dim, min_y: min_y + dim]
    roi = frame[min_y: min_y + dim, min_x: min_x + dim]
    if roi.shape[0] is not 0:
        cv2.imshow("RoI", roi)

    # roi = frame[min_y: min_y + dim, min_x: min_x + dim]

    if roi.shape[0] == dim and roi.shape[1] == dim:
        l, p = check_roi(roi)
        if l is not "OTHER":
            labels.append(l)
            percs.append(p)

    cv2.rectangle(frame, (min_x, min_y), (min_x - dim, min_y + dim), (0, 255, 0), 2)
    # roi = frame[min_x: min_x - dim, min_y: min_y + dim]
    roi = frame[min_y: min_y + dim, min_x: min_x - dim]

    if roi.shape[0] == dim and roi.shape[1] == dim:
        l, p = check_roi(roi)
        if l is not "OTHER":
            labels.append(l)
            percs.append(p)

    cv2.rectangle(frame, (min_x, min_y), (min_x + dim, min_y - dim), (0, 255, 0), 2)
    # roi = frame[min_x: min_x + dim, min_y: min_y - dim]
    roi = frame[min_y: min_y - dim, min_x: min_x + dim]

    if roi.shape[0] == dim and roi.shape[1] == dim:
        l, p = check_roi(roi)
        if l is not "OTHER":
            labels.append(l)
            percs.append(p)

    cv2.rectangle(frame, (min_x, min_y), (min_x - dim, min_y - dim), (0, 255, 0), 2)
    # roi = frame[min_x: min_x - dim, min_y: min_y - dim]
    roi = frame[min_y: min_y - dim, min_x: min_x - dim]

    if roi.shape[0] == dim and roi.shape[1] == dim:
        l, p = check_roi(roi)
        if l is not "OTHER":
            labels.append(l)
            percs.append(p)

    cv2.rectangle(frame, (min_x - dim / 2, min_y - dim / 2), (min_x + dim / 2, min_y + dim / 2), (255, 0, 0), 2)
    # roi = frame[min_x: min_x - dim, min_y: min_y - dim]
    roi = frame[min_y: min_y - dim, min_x: min_x - dim]

    if roi.shape[0] == dim and roi.shape[1] == dim:
        l, p = check_roi(roi)
        if l is not "OTHER":
            labels.append(l)
            percs.append(p)

    if percs.__len__() > 0:
        index = np.argmax(percs)
        test = percs[index]
        if test > 0.7:
            show_string = labels[index]
            percent = percs[index]



    print_sign_top_text(frame, show_string + ":" + str(percent))

    cv2.imshow("Motion detector", frame)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)

    # show the frame and record if the user presses a key
    key = cv2.waitKey(30)# & 0xFF
    if key == ord("q"):
        break

    # cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
