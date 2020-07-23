"""
	Virtual Writing Application using Aruco Markers
	Inspired by Virtual Pen by Taha Anwar
"""


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

class VirtualPen:
	def __init__(self, videoAddress, penID=8, eraserID=-1):
		self.videoAddress = videoAddress
		self.cap = cv2.VideoCapture(self.videoAddress)
		self.penID = penID
		self.eraserID = eraserID
		self.eraserThreshold = 38000

		# Setup Aruco Library
		self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
		self.parameters = cv2.aruco.DetectorParameters_create()

	def resizeFrame(self, frame, scale=0.2):
		frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
		return frame

	# Detect an aruco marker on the screen
	def detectTag(self, frame, singleTag=False):
		
		# Find corners of tag with ID = id
		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.parameters)
		allIds = np.squeeze(np.array(markerIds))
		index = np.squeeze(np.where(allIds==self.penID))

		if index >= 0:
			refPt = np.array(np.squeeze(markerCorners[index]), dtype=np.int32);
			refPt = refPt.reshape((-1,1,2))
			return refPt
		else:
			return np.array([])

	# Track its movement and draw with it
	def startDrawing(self, record = False):
		frameFound, frame = self.cap.read()
		frame = cv2.flip(frame, 1)
		if record:
			fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
			writeObj = cv2.VideoWriter("Output.mp4",fourcc, 30, (960,720), True)

		canvas = None
		clear = False

		# Initialize x1, y1 points
		x1, y1 = 0, 0
		# program_starts = time.time()

		while (frameFound):
			# now = time.time()
			# print(now - program_starts)
			# program_starts = now
			frameFound, frame = self.cap.read()

			if (frameFound == False):
				break

			if (self.videoAddress != 0):
				frame = self.resizeFrame(frame, 0.5)
			else:
				frame = self.resizeFrame(frame, 1.5)
			
			# Create a black canvas
			if canvas is None:
				canvas = np.zeros_like(frame)

			# Detect the tag
			corners = self.detectTag(frame)
			# print(corners)
			
			# Write on the canvas
			if len(corners)==0:
				x1, y1 = 0, 0
				continue
			else:
				x2, y2, w, h = cv2.boundingRect(corners)
				if x1==0 and y1==0:
					x1, y1 = int(x2+w/2),int(y2+h/2)
				else:
					canvas = cv2.line(canvas, (x1,y1), (int(x2+w/2),int(y2+h/2)), [255,0,0], 4)
				
				x1, y1 = int(x2+w/2),int(y2+h/2)

			# Delete if the user brought tag close to camera, else display the tag
			if (cv2.contourArea(corners) > self.eraserThreshold):
				# print(cv2.contourArea(corners))
				cv2.putText(canvas, 'Clearing Canvas', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5, cv2.LINE_AA)
				clear = True
			else:
				cv2.polylines(frame, [corners], True, (0,255,0), 3)

			_, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
			background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

			frame = cv2.add(background, canvas)
			frame = cv2.flip(frame,1) # Flip it while showing
			cv2.imshow("Canvas",frame)
			if record:	writeObj.write(frame)
		
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break

			# If clear screen requested, wipe out the canvas
			if clear == True:
				time.sleep(2)
				canvas = None
				clear = False

		cv2.destroyAllWindows()
		self.cap.release()


vp = VirtualPen(0,0)
vp.startDrawing(record=True)