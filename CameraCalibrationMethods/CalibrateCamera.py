"""
	Projecting 3d objects on a checkerboard, after calibrated camera
"""

import cv2
import glob
import numpy as np


# Camera Calibration Matrix
fx = 931.15874865
fy = 930.91937361
cx = 394.90314708
cy = 349.86798692
skew_gamma = 0

K = np.array([[fx, skew_gamma, cx], [0, fy, cy], [0, 0, 1]]) # Intrinsic Camera Calibration matrix

# Distortion coefficients
D = np.array([[-0.38871961, 1.37454222,-0.01041792, -0.00880581, -1.98486024]])


# Draw axis points(imgpts) over img given the corners of the checkerboard
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5) # x
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5) # y
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5) # z
    return img

# Termination Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# World Coordinates of Corners on the checkerboard
CHECKERBOARD = (6,9)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Axis points
axis = np.float32([[3,0,0], [0,3,0], [0, 0, -3]])

# Load previously saved data
with np.load('CameraCalibrationMatrices.npz') as X:
    K, D, R, T = [X[i] for i in ('arr_0','arr_1','arr_2','arr_3')]

# Project axes on each checkerboard image
for fname in glob.glob('PixelCheckboardImages/*.jpg'):
	img = cv2.imread(fname)

	scale_percent = 25 # percent of original size
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)

	dim = (width, height)
	img = cv2.resize(img, dim, cv2.INTER_AREA)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

	if ret:
		corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

		# Find the rotation and translation vectors.
		_, R, T, inliers = cv2.solvePnPRansac(objp, corners2, K, D)

		# project 3D points to image plane
		imgpts, jac = cv2.projectPoints(axis, R, T, K, D)

		img = draw(img,corners2,imgpts)
		cv2.imshow('img',img)
		
		k = cv2.waitKey(0) & 0xff
		if k == 's':
			cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()
