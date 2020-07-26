"""
	Python Application to reconstruct a 3d environment from 2d Images
	Currently, program is built to handle 2 images. Next goal is add it to handle multiple images

	Author: Sarvesh Thakur
"""

from construct3d import * ## Also import built-in libraries: cv2, numpy

def main():
	# Camera Calibration Matrix & # Distortion coefficients

	# Google Pixel 4 Camera
	fx = 931.15874865;	fy = 930.91937361
	cx = 394.90314708;	cy = 349.86798692
	skew_gamma = 0

	K_G = np.array([[fx, skew_gamma, cx], [0, fy, cy], [0, 0, 1]]) # Intrinsic Camera Calibration matrix
	d_G = np.array([[-0.38871961, 1.37454222,-0.01041792, -0.00880581, -1.98486024]])

	# Fountain Dataset Camera
	K = np.array([[2759.48/4, 0, 1520.69/4, 0, 2764.16/4,1006.81/4, 0, 0, 1]]).reshape(3, 3)
	d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)	# Already distortion rectified images

	# Load pair of images to perform SFM
	sfm = Reconstruct3D(K, d)
	imagePath = glob.glob("Dataset/FountainP11/*.png"); imagePath.sort()
	# sfm = Reconstruct3D(K_G, d_G)
	# imagePath = glob.glob("Dataset/PixelShot/*.jpg"); imagePath.sort()
	sfm.loadImages(imagePath[4], imagePath[5])

	# Track Feature Movement
	sfm.plotOpticalFlow()

	# Plot rectified images
	# sfm.plotAlignedImages()
	# cv2.imshow("Rectified Image",sfm.alignedImage)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	# Plot 3d Point Cloud
	sfm.plotPointCloud()



	# sfm.plot_point_cloud()
	sfm.create3DSurface()


if __name__ == "__main__":
	main()



