from collections import deque
import cv2
import imutils
import numpy as np


confidence_threshold = 0
suppresion_threshold = 1
input_width = 640
input_height = 640


'''def drawPred(classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


def get_outputs_names(nn):
    layersNames = nn.getLayerNames()
    return [layersNames[i[0] - 1] for i in nn.getUnconnectedOutLayers()]


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, suppresion_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

'''
scoreboard_established = False
first = False
first_x = 0
first_y = 0
track = True
detected = True
orig_frame = []
scoreboard = []
tracker = []
ball_color_lower = (10, 5, 0)
ball_color_higher = (100, 100, 100)
# ball_color_lower = (12, 38, 9)
# ball_color_higher = (25, 88, 78)
points = deque(maxlen=64)

vid = cv2.VideoCapture("2017 Golden State Warriors vs Cleveland Cavaliers Game 4.mov")
'''with open("coco.names", "rt") as n:
    classes = n.read().rstrip("\n").split("\n")
neural_network = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)'''

while True:
    # frame size : 720, 1280
    ret, frame = vid.read()
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
                '''blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_width, input_height), [0, 0, 0], 1, crop=False)
                neural_network.setInput(blob)
                temp = get_outputs_names(neural_network)
                outputs = neural_network.forward(temp)
                postprocess(frame, outputs)
                t, _ = neural_network.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
                cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))'''
                blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
                masked_frame = cv2.inRange(hsv_frame, ball_color_lower, ball_color_higher)
                masked_frame = cv2.erode(masked_frame, None, iterations=2)
                masked_frame = cv2.dilate(masked_frame, None, iterations=2)
                cv2.imshow("game", masked_frame)
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
    if not scoreboard_established:
        cv2.imshow('game', frame)
    # if detected:
        # cv2.imshow('game', frame)
    if cv2.waitKey(1) and 0xFF == ord('u'):
        break
