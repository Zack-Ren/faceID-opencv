import os
import numpy as np
import cv2
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

cur_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(img_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root)

			if not label in label_ids:
				label_ids[label] = cur_id
				cur_id += 1
			_id = label_ids[label]

			pil_image = Image.open(path).convert("L") #Grayscale

			size = (550, 550)	#scale training data to same size
			scaled_image = pil_image.resize(size, Image.ANTIALIAS)
			image_arr = np.array(scaled_image, "uint8") 

			faces = face_cascade.detectMultiScale(image_arr)

			for (x,y,w,h) in faces:
				faceROI = image_arr[y:y+h, x:x+h]
				x_train.append(faceROI)
				y_labels.append(_id)

#save labels
with open("labels.pickle", "wb") as f:
	pickle.dump(label_ids, f)

#save training data
face_recognizer.train(x_train, np.array(y_labels))
face_recognizer.save("train.yml")


print("Label:ID\n",label_ids)