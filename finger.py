import cv2
import time
import numpy as np

n_points = 22
contuor_pairs = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

treshold = 0.2
video = cv2.VideoCapture('video.mp4')
vid, frame = video.read()

f_width = frame.shape[1]
f_height = frame.shape[0]

aspect_ratio = f_width/f_height

height = 368
width = int(((aspect_ratio*height)*8)//8)

video_writer = cv2.VideoWriter('output.mkv', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))
files = cv2.dnn.readNetFromCaffe('pose_deploy.prototxt', 'pose_iter_102000.caffemodel')
k=0
while 1:
    k+=1
    t = time.time()
    vid, frame = video.read()
    framecopy = np.copy(frame)
    if not vid:
        cv2.waitKey(0)
        break

    blob = cv2.dnn.blobFromImage(frame, 1.0/255, (width, height), (0,0,0), swapRB=False, crop=False)

    files.setInput(blob)
    output = files.forward()
    print("forword = {}".format(time.time() - t))
    points = []

    for i in range(n_points):
        map = output[0, i, :, :]
        map = cv2.resize(map, (f_width, f_height))
        min_val, prob, min_loc, point = cv2.minMaxLoc(map)

        if prob > treshold :
            cv2.circle(framecopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(framecopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)
    
    for pair in contuor_pairs:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    print("Time take for frame = {}".format(time.time() - t))
    cv2.imshow('Outside Skeleton', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    print('Total = {}'.format(time.time() - t))
    print('---------------------------')
    video_writer.write(frame)
video_writer.release()