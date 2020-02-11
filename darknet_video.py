from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from ocr_plate import Plate
import random
import string

plate = Plate()

darknet.set_gpu(0)

def randomString(stringLength=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

val=0
def cvDrawBoxes(detections, img):
    global val
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin-2, ymin-2)
        pt2 = (xmax+2, ymax+2)
        
        # if (ymax-ymin) > 10 :
        #         # print(f'{detection}')
        #         val+=1
        #         crop = img[int(y)-int(h/2)-5:(int(y)+int(h/2))+5, int(x)-int(w/2)-5:(int(x)+int(w/2))+5]
        #         cv2.imwrite('frame/'+str(val)+'.png',crop)
                # if crop.size != 0:
                #     no_plat = plate.getText(crop)
                #     if len(no_plat) > 4:
                #         print(f"{no_plat}")
                #         cv2.imwrite('frame/'+str(no_plat)+'.png',crop)

        ########
        crop = img[int(y)-int(h/2)-5:(int(y)+int(h/2))+5, int(x)-int(w/2)-5:(int(x)+int(w/2))+5]
        # if crop.size != 0:
        #     no_plat = plate.getText(crop)
        #     print(f"{no_plat}")

        # #######
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        
        # plat  = str(detect_text(img[int(y):int(y)+int(h)-10, int(x):int(x)+int(w)]))
        # cv2.putText(img,
        #             detection[0].decode() +
        #             " [" + str(round(detection[1] * 100, 2)) + "] "+" no : "+plat,
        #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "cfg/yolov3-tiny_obj.cfg"
    weightPath = "backup/yolov3-tiny_final.weights"
    metaPath = "cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # cap = cv2.VideoCapture("vid7.mp4")
    cap = cv2.VideoCapture("rtsp://119.2.52.175:9997/s2")
    wi = int(cap.get(3))
    he = int(cap.get(4))
    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(wi,he,3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb,
                                   (wi,he),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_rgb.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.50)
        image = cvDrawBoxes(detections, frame_rgb)
        # image = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f'{1/(time.time()-prev_time)}')
        cv2.imshow('Demo', image)
        key = cv2.waitKey(50)
        if key == 27:
            print('Pressed Esc')
            break
    cap.release()
    # out.release()

if __name__ == "__main__":
    YOLO()
