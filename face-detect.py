from __future__ import print_function
import numpy as numpy
import cv2
import pickle
import argparse

USER_ID = "zack-ren"
LOCKED = True

def faceDetect(frame):
	#Turn output into mirror
	global LOCKED
	frame = cv2.flip(frame, 1)
	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.equalizeHist(frame_gray)
	faces = face_cascade.detectMultiScale(frame_gray)
	font = cv2.FONT_HERSHEY_SIMPLEX
	WHITE = (255,255,255)
	cv2.putText(frame, "Looking for you",(200,250), font, 1, (0,0,255), 1, cv2.LINE_AA)
	if LOCKED:
		for (x,y,w,h) in faces:
			frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
			faceROI = frame_gray[y:y+h, x:x+w]

			#Facial recognization, person identification
			_id, conf = face_recognizer.predict(faceROI)
			if conf >= 90:
				cv2.putText(frame, labels[_id], (x,y), font, 0.5, WHITE, 2, cv2.LINE_AA)
				if labels[_id] == USER_ID:
					img_item = "output.png"
					cv2.imwrite(img_item, faceROI)
					LOCKED = False

			#Detect eyes
			eyes = eyes_cascade.detectMultiScale(faceROI)
			for (x2,y2,w2,h2) in eyes:
				eye_center = (x + x2 + w2//2, y + y2 + h2//2)
				radius = int(round((w2 + h2)*0.25))
				frame = cv2.circle(frame,eye_center,radius, (255,0,0),2)

	if not LOCKED:
		frame = cv2.rectangle(frame, (0,0),(700, 500),(0,0,0),-1)
		text = 'Welcome back, '+ USER_ID
		cv2.putText(frame, text, (125, 250), font, 1, WHITE, 1, cv2.LINE_AA)

	cv2.imshow('Facial Detection', frame)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("train.yml")

#open saved labels
labels = {}
with open("labels.pickle", "rb") as f:
	labels = pickle.load(f)
	labels = {v:k for k,v in labels.items()}

parser = argparse.ArgumentParser(description='Haar Cascade Facial Detection')
parser.add_argument('--face_cascade', help='path to face cascade', default='cascades/data/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='cascades/data/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)

args = parser.parse_args()
face_cascade_arg = args.face_cascade
eyes_cascade_arg = args.eyes_cascade
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_arg)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_arg)):
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = args.camera
cap = cv2.VideoCapture(camera_device)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    faceDetect(frame)
    if cv2.waitKey(10) == 27:
    	break
