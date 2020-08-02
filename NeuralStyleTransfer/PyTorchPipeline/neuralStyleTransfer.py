"""
	Transfer style to new images from paintings and other styles.
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

# Available model names
modelNames = ['feathers', 'candy', 'composition_vii', 'udnie', 'the_wave', 'the_scream',
				'mosaic', 'la_muse', 'starry_night']

# Initialize the DNN module

def initializeModel(styleType='candy', useGPU=None):
	global net
	global style
	style = styleType

	if styleType in modelNames:
		model = '../StyleModels/' + styleType + '.t7'
		net = cv2.dnn.readNetFromTorch(model)

		# If gpu preferred
		if useGPU == 'cuda':
			net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		elif useGPU == 'opencl':
			net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

	else:
		print("Model not Found")
		sys.exit()


def transferStyle(image = [], userCam=False, returnData=False):

	if np.size(image) == 0:
		if userCam:
			cap = cv2.VideoCapture(0, cv2.CP_DSHOW)
			ret, frame = cap.read()
			if ret:
				image = cv2.flip(frame, 1)
				cap.release()
			else:
				print("Can't open camera")
				sys.exit()
	# elif type(image) == str:
	# 	image = cv2.imread(image)


	# Pre Processing
	R, G, B = 103.939, 116.779, 123.680
	blob = cv2.dnn.blobFromImage(image, 1.0, (image.shape[1], image.shape[0]), [R,G,B], swapRB=False, crop=False)
	net.setInput(blob)

	# Forward Pass
	output = net.forward()

	# Post Processing
	finalOutput = output.reshape((3, output.shape[2], output.shape[3])).copy()
	finalOutput[0] += R; finalOutput[1] += G; finalOutput[2] += B
	finalOutput = finalOutput.transpose(1,2,0)
	postProcessedImage = np.clip(finalOutput, 0,255)
	finalStyledImage = postProcessedImage.astype('uint8')

	if returnData:
		return finalStyledImage
	else:
		plt.imshow(finalStyledImage[:,:,::-1])
		plt.axis('off')
		plt.show()
		name = "../testImages/Output" + style + ".jpg"
		cv2.imwrite(name, finalStyledImage)

initializeModel('feathers')
image = cv2.imread("../testImages/AchiPhoto.jpg")
transferStyle(image)