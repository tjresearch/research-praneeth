from collections import deque
import cv2
import imutils
import numpy as np


confidence_threshold = 0
suppresion_threshold = 1
input_width = 640
input_height = 640
scoreboard_established = False
first = False
first_x = 0
first_y = 0
track = True
detected = True
orig_frame = []
scoreboard = []
tracker = []
ball_color_lower = (0, 93, 76)
ball_color_higher = (21, 174, 130)
points = deque(maxlen=64)

vid = cv2.VideoCapture("2017 Golden State Warriors vs Cleveland Cavaliers Game 4.mov")

while True:
    # frame size : 720, 1280
    ret, frame = vid.read()
    # frame size : 338, 600
    imutils.resize(frame, width=600)
    if not ret:
        print("Video over")
        break
    if scoreboard_established:
        (success, boxes) = tracker.update(frame)
        if not success:
            print("Not Detected")
            tracker = cv2.MultiTracker_create()
            temp_tracker = cv2.TrackerMOSSE_create()
            tracker.add(temp_tracker, orig_frame, scoreboard)
            frame = imutils.resize(frame, width=1280)
            cv2.imshow('game', frame)
            cv2.waitKey(1)
            continue
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            if first:
                first_x = x
                first_y = y
                first = False
            if (first_x - 10 <= x <= first_x + 10) and (first_y - 10 <= y <= first_y + 10):
                print("Detected")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
                masked_frame = cv2.inRange(hsv_frame, ball_color_lower, ball_color_higher)
                masked_frame = cv2.erode(masked_frame, None, iterations=2)
                masked_frame = cv2.dilate(masked_frame, None, iterations=2)
                contours = cv2.findContours(masked_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                center = None
                circle_list = []
                if len(contours) == 0:
                    contours_found = False
                    print("mask did not find ball")
                else:
                    for x in contours:
                        approx = cv2.approxPolyDP(x, 0.01 * cv2.arcLength(x, True), True)
                        if 10 <= len(approx) <= 12:
                            circle_list.append(x)
                    min_error = float("inf")
                    if len(circle_list) > 0:
                        min_contour = circle_list[0]
                    is_contour = False
                    for c in circle_list:
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        '''if x < 100 or x > 500 or y < 40 or y > 300:
                            continue'''
                        error = abs(978.3506508 - cv2.contourArea(c))
                        if error < min_error:
                            min_error = error
                            min_contour = c.copy()
                            is_contour = True
                    if is_contour:
                        ((x, y), radius) = cv2.minEnclosingCircle(min_contour)
                        moments = cv2.moments(min_error)
                        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.imshow("game", frame)
            else:
                detected = False
                print("Not Detected")
                tracker = cv2.MultiTracker_create()
                temp_tracker = cv2.TrackerMOSSE_create()
                tracker.add(temp_tracker, orig_frame, scoreboard)
    if cv2.waitKey(3) == ord("j") and not scoreboard_established:
        scoreboard_established = True
        first = True
        temp_tracker = cv2.TrackerMOSSE_create()
        tracker = cv2.MultiTracker_create()
        scoreboard = cv2.selectROI('game', frame, fromCenter=False, showCrosshair=True)
        orig_frame = frame.copy()
        tracker.add(temp_tracker, orig_frame, scoreboard)
    '''if not scoreboard_established:
        blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        masked_frame = cv2.inRange(hsv_frame, ball_color_lower, ball_color_higher)
        masked_frame = cv2.erode(masked_frame, None, iterations=2)
        masked_frame = cv2.dilate(masked_frame, None, iterations=2)
        contours = cv2.findContours(masked_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        center = None
        circle_list = []
        if len(contours) == 0:
            contours_found = False
            print("mask did not find ball")
        else:
            for x in contours:
                approx = cv2.approxPolyDP(x, 0.01 * cv2.arcLength(x, True), True)
                if 10 <= len(approx) <= 12:
                    circle_list.append(x)
            for c in circle_list:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
            cv2.drawContours(frame, circle_list, -1, (255, 0, 0), 2)'''
    cv2.imshow('game', frame)
    if cv2.waitKey(1) and 0xFF == ord('u'):
        break
