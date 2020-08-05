from pyimagesearch.tempimage import TempImage
from picamera.array import PiRGBArray
from picamera import PiCamera
import quickstart as qs
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import os

name_folder = 'Surveillance_OpenCV'
mimeType = "mimeType = 'application/vnd.google-apps.folder' and name = '{0}'".format(name_folder)

ap = argparse.ArgumentParser()
ap.add_argument("-c","--conf",required=True,
                help="path to JSON configuration file")
args = vars(ap.parse_args())

warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

print('[INFO] API Google Drive starting...')
service = qs.main()

camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size = tuple(conf["resolution"]))

print("[INFO] warmimg up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0


flag, id_folder = qs.query_files(service,mimeType)
if flag:
    print('Folder exists')
else:
    id_folder = qs.create_folder(service,name_folder)
    print('{} folder created'.format(name_folder))


for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    frame = f.array
    timestamp = datetime.datetime.now()
    text = "Unoccupied"

    frame = imutils.resize(frame,width=700)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if avg is None:
        print("[INFO] starting backgroud model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue

    cv2.accumulateWeighted(gray,avg,0.5)
    frameDelta = cv2.absdiff(gray,cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta,conf["delta_thresh"],255,
                            cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh,None,iterations=2)

    cv2.imshow('frameDelta',frameDelta)
    cv2.imshow('th',thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        text = "Occupied"

    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if text == "Occupied":
        if (timestamp -lastUploaded).seconds >= conf["min_upload_seconds"]:
            motionCounter += 1

        if motionCounter >= conf["min_motion_frames"]:

            #t = TempImage()
            #cv2.imwrite(t.path,frame)

            path_file = '/home/pi/Desktop/'
            path = '{base_path}{timestamp}.jpg'.format(
                    base_path = path_file,timestamp=ts)
            print(path)
            cv2.imwrite(path,frame)
            cv2.imshow('Frame capturated',frame)
            #byteArray = open(t.path,'rb').read()
            #image = Image.open(io.BytesIO(byteArray))
            #image.save(path_file+'frame.jpg')
            qs.file_in_folder(service,path,id_folder,path_file)
            os.remove(path)
            #t.cleanup()
            lastUploaded = timestamp
            motionCounter = 0
    else:
        motionCounter = 0

    if conf["show_video"]:
        cv2.imshow("Security feed",frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    rawCapture.truncate(0)

