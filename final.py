
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2
import pygame

eyePoint = 0.3
continousFrames = 38
blinkContinousframes=7
alarm = False
count=0


data="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(data)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
def sound_alarm():
	pygame.mixer.init()
	pygame.mixer.music.load("C:\\Users\\Chinni\\Desktop\\raspberry\\ala.ogg")
	pygame.mixer.music.play()
def formula(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	eyeDifference = (A + B) / (2.0 * C)
	return eyeDifference



cap = cv2.VideoCapture(0)

while True:
	
	video = cap.read()[1]
	gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
	det = detector(gray, 0)
	for i in det:
		shape = predictor(gray,i)
		shape = face_utils.shape_to_np(shape)
		leye = shape[lStart:lEnd]
		reye = shape[rStart:rEnd]
		leyeDifference = formula(leye)
		reyeDifference = formula(reye)
		eyeDifference1 = (leyeDifference + reyeDifference) / 2.0
		leyeborder = cv2.convexHull(leye)
		reyeborder = cv2.convexHull(reye)
		cv2.drawContours(video, [leyeborder], -1, (0, 255, 0), 1)
		cv2.drawContours(video, [reyeborder], -1, (0, 255, 0), 1)
		cv2.circle(video,(335,263), 360, (0,0,0), 270)

		if eyeDifference1 < eyePoint:
			count += 1
			if count >=continousFrames:
				if not alarm:
					alarm = True					
					t = Thread(target=sound_alarm)
						
					t.deamon = True
					t.start()
				cv2.putText(video, "You Are Sleeping", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(video, "Eyes close".format(video), (10, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			elif count >= blinkContinousframes:
				cv2.putText(video, "YOU JUST BLINKED", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

				
		else:
			count = 0
			alarm = False
			cv2.putText(video, "THE DRIVER IS AWAKE", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 255, 0), 2)
			cv2.putText(video, "Eyes Open ", (10, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
		
	cv2.imshow("OUTPUT", video)
	k=cv2.waitKey(1)
	if k==27:
		break


cv2.destroyAllWindows()
