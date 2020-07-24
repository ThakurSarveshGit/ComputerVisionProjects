"""
	Create Panorama from Images using OpenCV, Python.
	Author: Sarvesh Thakur
"""


import cv2
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Feature Extraction
orb = cv2.ORB_create(5000)

allPhotos = glob.glob("Data/Photos/*.jpg")
allPhotos.sort()

image1 = cv2.imread(allPhotos[0], 1)
image1 = cv2.resize(image1, (0,0), fx=0.2, fy=0.2)
image2 = cv2.imread(allPhotos[1], 1)
image2 = cv2.resize(image2, (0,0), fx=0.2, fy=0.2)
image3 = cv2.imread(allPhotos[2], 1)
image3 = cv2.resize(image3, (0,0), fx=0.2, fy=0.2)

# Find keypoints, descriptors using ORB
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)
kp3, des3 = orb.detectAndCompute(image3, None)

# Step 2: Feature Matching

bf = cv2.BFMatcher(cv2.NORM_HAMMING,  crossCheck=True)
matches = bf.match(des1, des2)
matches.sort(key = lambda x:x.distance)

match_percent = 0.99
good_matches = matches[:int(len(matches)*match_percent)]

matches_img = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=2)

# Step 3: Homography

# Getting the x, y coordinates of the best matces in the right format
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
des_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# Get Homography
H, _ = cv2.findHomography(des_pts, src_pts, cv2.RANSAC)
print(H)

# Align Image2 with Image 1
h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]

alignedImage = cv2.warpPerspective(image2, H, (w1+w2, h2))

# Step 4: Stich the Image
stitched = alignedImage.copy()
stitched[0:h1, 0:w1] = image1

# stitcher = cv2.Stitcher.create()
# cvStitched = stitcher.stitch([image1, image2])
# print(cv2.Stitcher_OK)


cv2.imshow("Original Two Image", np.hstack((image1,image2)))
cv2.imshow("alignedImage", alignedImage)
cv2.imshow("Panorama", stitched)
cv2.imwrite("Result.jpg", stitched)
cv2.waitKey(0)
cv2.destroyAllWindows()
