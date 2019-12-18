from collections import deque
import cv2
import imutils
import math


def calculate_distance(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


confidence_threshold = 0
suppresion_threshold = 1
input_width = 640
input_height = 640
scoreboard_established = False
first = False
first_x = 0
first_y = 0
track = True
orig_frame = []
scoreboard = []
tracker = []
ball_color_lower = (0, 93, 76)
ball_color_higher = (21, 174, 130)
points = deque(maxlen=64)
prev_circles = []

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
                    circle_list.sort(key=lambda circle: abs(978.3506508 - cv2.contourArea(circle)))
                    '''for c in circle_list:
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        error = abs(978.3506508 - cv2.contourArea(c))
                        if error < min_error:
                            min_error = error
                            min_contour = c.copy()'''
                    if len(circle_list) > 10:
                        circle_list = circle_list[:10]
                    cur_circles = []
                    for c in circle_list:
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        moments = cv2.moments(c)
                        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                        if 240 <= int(y) <= 650 and 10 <= int(radius) <= 16:
                            cur_circles.append([x, y, radius])
                        '''cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)'''
                    if len(prev_circles) != 0 and len(cur_circles) != 0:
                        print("prev_circles exists")
                        min_distance = float("inf")
                        min_circle = cur_circles[0]
                        for a in cur_circles:
                            for b in prev_circles:
                                distance = calculate_distance(a[0], b[0], a[1], b[1])
                                if distance < min_distance and distance <= 7:
                                    min_distance = distance
                                    min_circle = a.copy()
                        cv2.circle(frame, (int(min_circle[0]), int(min_circle[1])), int(min_circle[2]), (0, 255, 255), 2)
                        cv2.circle(frame, (int(min_circle[0]), int(min_circle[1])), 5, (0, 0, 255), -1)
                    prev_circles = cur_circles.copy()
                cv2.imshow("game", frame)
            else:
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
    cv2.imshow('game', frame)
    if cv2.waitKey(1) and 0xFF == ord('u'):
        break
