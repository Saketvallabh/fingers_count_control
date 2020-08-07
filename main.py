#import pakage for thr code
import numpy as np
import cv2
import math
import pyautogui

#open webcam for capturing video
capture = cv2.VideoCapture(0)

def minimize_all():
	pyautogui.keyDown('winleft')
	pyautogui.press('m')
	pyautogui.keyUp('winleft')

def alt_tab():
	pyautogui.keyDown('altleft')
	pyautogui.press('tab')
	pyautogui.keyUp('altleft')

def alt_shift_tab():
	pyautogui.keyDown('altleft')
	pyautogui.keyDown('shiftleft')
	pyautogui.press('tab')
	pyautogui.keyUp('shiftleft')
	pyautogui.keyUp('altleft')

#creating while loop that will run untill the webcam in on
while capture.isOpened():
	ret,frames = capture.read()

	#getting hand data in rectangle sun window
	cv2.rectangle(frames, (90,90),(300,300), (0,255,0),0)
	crop_frame = frames[90:300, 90:300]

	#applying gaussian blur
	blur = cv2.GaussianBlur(crop_frame, (3,3),0)

	#changing color- space from BGR to HSV
	hsv =cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	#create a binary image in which skin will be white and rest will be black
	mask = cv2.inRange(hsv, np.array([2,0,0]), np.array([20, 255, 255]))

	#kernel for morphological tranformation
	kernel = np.ones((5,5))

	# applying morphological tranformations to filtre out the background noise
	dilated = cv2.dilate(mask, kernel, iterations=1)
	eroded = cv2.erode(dilated, kernel, iterations=1)

	#applying gaussian blur and threshold
	filtered = cv2.GaussianBlur(eroded, (3,3), 0)
	ret, thres = cv2.threshold(filtered, 127, 255, 0)

	#show the threshold image
	cv2.imshow("thresholded frame", thres)

	#finding contours
	contours, hierarchy = cv2.findContours(thres.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	try:
		# finding contour with maximum area
		contour = max(contours, key=lambda x: cv2.contourArea(x))

		#create bounded rectangle around the contour
		x, y, w, h = cv2.boundingRect(contour)
		cv2.rectangle(crop_frame, (x, y), (x+w,y+h), (0,0,255), 0)

		#finding convex hull
		hull = cv2.convexHull(contour)

		#draw contour
		drawcontour = np.zeros(crop_frame.shape, np.uint8)
		cv2.drawContours(drawcontour, [contour], -1, (0, 255, 0), 0)
		cv2.drawContours(drawcontour, [hull], -1, (0, 0, 255), 0)

		#finding convexity defects
		hull = cv2.convexHull(contour, returnPoints=False)
		defects = cv2.convexityDefects(contour, hull)

		#we are using cosine formula for finding the angle of far point from the start point and end point
		count_defects = 0

		for i in range(defects.shape[0]):
			s,e,f,d =defects[i,0]
			start = tuple(contour[s][0])
			end = tuple(contour[e][0])
			far = tuple(contour[f][0])

			a= math.sqrt((end[0]-start[0]) ** 2+ (end[1]-start[1]) ** 2)
			b= math.sqrt((far[0]-start[0]) ** 2 + (far[1]-start[1]) ** 2)
			c= math.sqrt((end[0]-far[0]) ** 2 +(end[1]-far[1]) ** 2)
			angle = (math.acos((b ** 2 + c ** 2 -a ** 2)/(2*b*c))*180)/3.14

			#if angle is greater than 90 draw a point cicle  at the far point
			if angle <=90:
				count_defects += 1
				cv2.circle(crop_frame, far, 1, [0,255,0], -1)

			cv2.line(crop_frame, start, end, [255,0,0], 2)

		#printing the number of fingers
		if count_defects ==0:
			cv2.putText(frames, "ONE", (40,40), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
		elif count_defects ==1:
			cv2.putText(frames, "TWO", (40,40), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
		elif count_defects ==2:
			cv2.putText(frames, "THREE", (40,40), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
			alt_tab()
		elif count_defects ==3:
			cv2.putText(frames, "FOUR", (40,40), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
			alt_shift_tab()
		elif count_defects ==4:
			cv2.putText(frames, "FIVE", (40,40), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
			minimize_all()
		else:
			pass
	except:
		pass
	drawcontour = np.zeros(crop_frame.shape, np.uint8)
	#show required images
	cv2.imshow("gesture", frames)
	all_image = np.hstack((drawcontour, crop_frame))
	cv2.imshow('contours', all_image)

	#close the camera if q id pressed
	if cv2.waitKey(1)== ord('q'):
		break

capture.release()
cv2.destroyAllWindows()