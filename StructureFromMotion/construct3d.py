"""
	Header class to perform 3D Reconstruction using Structure from Motion

	Operation Steps:
		1. Feature Tracking using Optic Flow
		2. Image Rectification
		3. Triangulation
		4. 3D Point Cloud Visualization
"""

import cv2
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

class Reconstruct3D:

	def __init__(self, K, d):
		self.K = K
		self.K_inv = np.linalg.inv(K)
		self.d = d

	def preProcessImage(self, image, downScale=True):
		# Preprocess if required
		
		# Downsize
		minSize = 600
		if downScale and image.shape[1] > minSize:
			while (image.shape[1] > 2*minSize):
				image = cv2.pyrDown(image)
			# print("Image DownSized")
		
		# Undistort
		image = cv2.undistort(image, self.K, self.d)

		# print("[Reconstruct3D]:[preProcessImage]: Image preprocessed successfully")
		return image

	def loadImages(self, path1, path2, downScale=True):
		# Load images, assumed color images
		self.image1 = cv2.imread(path1, cv2.CV_8UC3)
		self.image2 = cv2.imread(path2, cv2.CV_8UC3)
		
		# Preprocess images, if required
		self.image1 = self.preProcessImage(self.image1, downScale)
		self.image2 = self.preProcessImage(self.image2, downScale)

		print("[Reconstruct3D]:[loadImages]: Images read successfully")

	def extractKeypointsFlow_(self):
		
		# FAST Keypoint Detector
		fast = cv2.FastFeatureDetector_create()
		firstKeyPoints = fast.detect(self.image1, None)

		# Calculate Optical Flow
		firstKeyList = [i.pt for i in firstKeyPoints]
		firstKeyArr = np.array(firstKeyList).astype(np.float32)

		#Find final position of feature points
		secondKeyArr, status, error = cv2.calcOpticalFlowPyrLK(self.image1, self.image2, firstKeyArr, None)

		condition = (status == 1)*(error < 5)
		concat = np.concatenate((condition, condition), axis=1)

		self.match_pts1 = firstKeyArr[concat].reshape(-1, 2)
		self.match_pts2 = secondKeyArr[concat].reshape(-1, 2)

		# print(self.match_pts1, self.match_pts2)
		print("[Reconstruct3D]:[extractKeypointsFlow_]: Optical Flow Calculated")

	def plotOpticalFlow(self):
		
		# Calculate optical flow
		self.extractKeypointsFlow_()

		image = self.image1.copy()

		for i in range(len(self.match_pts1)):
			cv2.line(image, tuple(self.match_pts1[i]), tuple(self.match_pts2[i]), color = (255,0,0))
			theta = np.arctan2(self.match_pts2[i][1]-self.match_pts1[i][1],
							self.match_pts2[i][0]-self.match_pts1[i][0])
			cv2.line(image, tuple(self.match_pts2[i]), (np.int(self.match_pts2[i][0] - 6*np.cos(theta-np.pi/4)), np.int(self.match_pts2[i][1] - 6*np.sin(theta-np.pi/4))),
														color=(255,0,0))
			cv2.line(image, tuple(self.match_pts2[i]), (np.int(self.match_pts2[i][0] - 6*np.cos(theta+np.pi/4)), np.int(self.match_pts2[i][1] - 6*np.sin(theta+np.pi/4))),
														color=(255,0,0))

		self.opticalFlowPlotImage = image

	def findFundamentalMatrix_(self):
		self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1, self.match_pts2, cv2.FM_RANSAC, 0.1, 0.99)


	def findEssentialMatrix_(self):
		self.E = self.K.T.dot(self.F).dot(self.K)


	def checkPointsInCameraFront_(self, firstPoints, secondPoints, R, T):

		for i, j in zip(firstPoints, secondPoints):
			firstZ = np.dot(R[0,:] - j[0]*R[2, :], T)/np.dot(R[0,:] - j[0]*R[2,:], j)
			first3DPoints = np.array([i[0]*firstZ, j[0]*firstZ, firstZ])
			second3DPoints = np.dot(R.T, first3DPoints) - np.dot(R.T, T)

			if first3DPoints[2] < 0  or second3DPoints[2] < 0:
				return False

		return True


	def findCameraMatrices_(self):
		U, S, Vt = np.linalg.svd(self.E)
		W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3,3)

		# Iterate over all point correspondences used in the estimation of F Matrix
		firstInliers = []; secondInliers = []
		for i in range(len(self.Fmask)):
			if self.Fmask[i]:
				firstInliers.append(self.K_inv.dot([self.match_pts1[i][0],
													self.match_pts1[i][1],
													1.0]))
				secondInliers.append(self.K_inv.dot([self.match_pts2[i][0],
													self.match_pts2[i][1],
													1.0]))

		# Find Camera Matrix that projects all point only in front of it

		# First Choice: R = U*W*Vt; T = +U3
		R = U.dot(W).dot(Vt)
		T = U[:,2]
		checkFront = self.checkPointsInCameraFront_(firstInliers, secondInliers, R, T)

		if checkFront == False:
			# Second Choice: R = U*W*Vt; T= -U3
			T = -T
			checkFront = self.checkPointsInCameraFront_(firstInliers, secondInliers, R, T)

		if checkFront == False:
			# Third Choice: R = U*Wt*Vt; T = +U3
			R = U.dot(W.T).dot(Vt)
			T = U[:,2]
			checkFront = self.checkPointsInCameraFront_(firstInliers, secondInliers, R, T)

		if checkFront == False:
			# Fourth Choice: R = U*Wt*Vt; T = -U3
			T = -U[:,2]

		self.matchInliers1 = firstInliers
		self.matchInliers2 = secondInliers

		self.Rt1 = np.hstack((np.eye(3), np.zeros((3,1))))	# For pose 1 of camera
		self.Rt2 = np.hstack((R, T.reshape(3,1)))	# For pose 2 of camera

		print("[Reconstruct3D]:[findCameraMatrices_]: Transforamtion1 : {},\nTransformation2: {}".format(self.Rt1, self.Rt2))


	def plotAlignedImages(self):
		# Plot images using Warp Perspective instead of stereo correction;
		# Had hard time making Stereo Function work
		
		self.extractKeypointsFlow_() # Track Features using Optical Flow
		self.findFundamentalMatrix_() # Find Fundamental Matrix
		self.findEssentialMatrix_() # Find Essential Matrix
		self.findCameraMatrices_() # Find Camera Matrices

		H, _ = cv2.findHomography(self.match_pts2, self.match_pts1, cv2.RANSAC)
		# print("H: ",H)
		
		h1, w1 = self.image1.shape[:2]
		h2, w2 = self.image2.shape[:2]

		self.alignedImage = cv2.warpPerspective(self.image2, H, (w1+w2, h2))
		self.alignedImage[:,self.image2.shape[1]:] = self.image1

		for i in range(20, self.alignedImage.shape[0], 25):
			cv2.line(self.alignedImage, (0, i), (self.alignedImage.shape[1], i), (255, 0, 0))


	def plotPointCloud(self, matplotlibPlot=False):
		
		# Plot Point Cloud - Complete Procedure

		self.extractKeypointsFlow_() # Track Features using Optical Flow
		self.findFundamentalMatrix_() # Find Fundamental Matrix
		self.findEssentialMatrix_() # Find Essential Matrix
		self.findCameraMatrices_() # Find Camera Matrices

		# Triangulate Points
		firstInliers = np.array(self.matchInliers1).reshape(-1,3)[:, :2]
		secondInliers = np.array(self.matchInliers2).reshape(-1,3)[:, :2]
		points4D = cv2.triangulatePoints(self.Rt1, self.Rt2, firstInliers.T, secondInliers.T).T
		# print(points4D.shape)

		# Convert from Homogeneous Coordinates to 3D
		self.points3D = points4D[:, :3]/np.repeat(points4D[:,3],3).reshape(-1,3)
		print("No. of Points to be projected back in World: {}".format(self.points3D.shape[0]))

		# Plot with matplotlib
		self.Xs = self.points3D[:, 0]
		self.Ys = self.points3D[:, 1]
		self.Zs = self.points3D[:, 2]

		if matplotlibPlot:
			figure = plt.figure()
			axis = figure.add_subplot(111, projection="3d")
			axis.scatter(self.Xs, self.Ys, self.Zs, c='b', marker='.')
			axis.set_xlabel('X')
			axis.set_ylabel('Y')
			axis.set_zlabel('Z')
			plt.title('3D point Cloud')
			plt.show()

	def lod_mesh_export(self, mesh, lods, extension, path):
		mesh_lods = {}
		for i in lods:
			mesh_lod = mesh.simplify_quadric_decimation(i)
			o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
			mesh_lods[i] = mesh_lod
		print("Generation of " + str(i) + "loD successful")
		return mesh_lods

	def create3DSurface(self, matplotlibPlot = False):
		"""
			Create a 3D surface using Open3D
		"""
		
		# Calculate Point Cloud
		self.plotPointCloud(False)
		
		# write 3D points to a file
		np.savetxt('test.txt', np.asarray(self.points3D))

		# Load point cloud
		pointCloud = np.loadtxt('test.txt')
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(pointCloud)
		pcd.estimate_normals()
		
		# Visualize the point Cloud
		o3d.visualization.draw_geometries([pcd])

		# Create a 3D Surface

		# Poisson Method
		poissonMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

		# Crop extra layers
		bbox = pcd.get_axis_aligned_bounding_box()
		p_mesh_crop = poissonMesh.crop(bbox)

		# Visualize the surface
		o3d.io.write_triangle_mesh("fountainP11_p_mesh_c.ply", p_mesh_crop)

		# my_lods = self.lod_mesh_export(p_mesh_crop, [8000,800,300], ".ply", "")


		# o3d.visualization.draw_geometries([my_lods[800]])






