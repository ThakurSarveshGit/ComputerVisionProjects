# Computer Vision Projects - Classical and Modern Approaches
Contains Computer Vision projects I implemented out of interest. 

Projects aimed at:
1. Classical Computer Vision Techniques
2. Advanced Computer Vision Techniques
3. Computer Vision using ML/DL

# Classical Computer Vision Projects

## Project Zero: A Computer Vision Class for complete SLAM solution.  
- [ ] Completed

## Project One: Virtual Pen and Eraser  
- [x] Completed

I recently came across posts on linkedin, where people were drawing virtually on the screen, originally posted by Mr. Taha Anwar. So, I thought why not try it on my own! Though the original version works on color thresholding to extract the pen from the scene, I had a hard time(& changing light intensities in my background) finding a right range. Hence, I added an aruco marker(tag id: 0) to act as a pen. Feel free to use this version and build your version upon it. A couple of other id markers can work for various colors and even eraser.
<p align="center">
<img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/VirtualPen%26Eraser/GIF_VirtualPen_2_Small.gif" alt="OK?OK" width="250"/>
</p>

## Project Two: Panaroma Creator  
- [x] Completed

In this mini-mini-project, I have build a custom panaroma script using Feature Matching through ORB algorithm. This is followed by Image Alignment after finding the Homogenous Transformation between the two images and applying perspective transformation to one of the image.
<p align="center">
<img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/CreatePanaroma/Result.jpg" alt="PanoramaResult" width="250"/>
</p>

## Project Three: Buoy Detection using Gaussian Mixture Models

This project introduced the concept of color segmentation using Gaussian Mixture Models and Expectation Maximization techniques. The video sequence provided has been captured underwater and shows three buoys of different colors, namely yellow, orange and green. They are almost circular in shape and are distinctly colored. However, conventional segmentation techniques involving color thresholding will not work well in such an environment, since noise and varying light intensities will render any hard-coded thresholds ineffective. In such a scenario, we “learn” the color distributions of the buoys and use that learned model to segment them. This project required us to obtain a tight segmentation of each buoy for the entire video sequence by applying a tight contour (in the respective color of the buoy being segmented) around each buoy.
<p align="center">
  <img src="https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/blob/master/BuoyDetectionGMM/buoyDetection.gif" alt="Underwater Buoy Detection using Gaussian Mixture Models(GMMs)"  width="250"/>
</p>

## [Project Four: Lane Detection using Sliding Windows](https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/tree/master/AdvanceLaneDetection)

In this project we aim to do simple Lane Detection to mimic Lane Departure Warning systems used in SelfDriving Cars.  We are provided with two video sequences(one in normal lighting and other one in changing light intensities), taken from a self driving car.  Our task was to design an algorithm to detect lanes on the road,as well as estimate the road curvature to predict car turns.

<p align="center">
  <img src="https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/blob/master/AdvanceLaneDetection/laneDetection.gif" alt="Advance Lane Detection for Autonomous Cars"  width="250"/>
</p>

# Advanced Computer Vision Techniques

## Project One: Structure from Motion  
- [x] Completed

In this project, I performed structure from motion(a.k.a structure and motion) on an image dataset. The project was extensive and it required a good hand in geometrical transformation techniques. Overall, I am happy(somewhat) to finish it but there are a great amount of modifications I see possible to make it better. I update this project and make it better soon. Here is my course of action:

- [x] Camera Calibration for custom dataset.  
- [x] Feature Tracking using Optic Flow.  
- [x] Image Rectification.  
- [x] Triangulation.  
- [x] 3D Point Cloud Generation.

<img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/StructureFromMotion/TopView.png" alt="SFM" width="250"/>

## Project Two: Augmented Reality  
- [ ] Completed

In this project, I am placing multiple 3d objects(solid ones) on a flat surface. Coming Soon!

# . . .


# Computer Vision through Deep Learning Projects

## Project One: Facial Keypoints Recognition  
- [x] Completed

In this project, I combined my knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face! Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications.

The project is broken down into four main parts:

[Notebook 1](FacialKeypointRecognition/1_LoadAndVisualizeData.ipynb): Loading and Visualizing the Facial Keypoint Data  
[Notebook 2](FacialKeypointRecognition/2_Define_TheNetworkArchitecture.ipynb): Defining and Training a Convolutional Neural Network (CNN) to predict Facial Keypoints  
[Notebook 3](FacialKeypointRecognition/3_FacialKeypointDetectionCompletePipeline.ipynb): Facial Keypoint Detection Using Haar Cascades and our trained CNN.  
[Notebook 4](FacialKeypointRecognition/4_ApplicationsKeypoints.ipynb): Applications using Facial Keypoints


## Project Two: Traffic Sign Detection

The focus of this project is divided into two pipelines, Traffic sign detection and traffic sign classification. Traffic sign detection is the process of forming a bounding box around a traffic sign, so that the region of interest can be cropped and given to the classifier for sign classification. HSV feature and MSER features are tested for robust sign detection. Model is trained using Support Vector Machines.

<p align="center">
  <img src="https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/blob/master/TrafficSignsDetection/GIF_TrafficSignDetect.gif" alt="Homography and Pose Estimation" width="250"/>
</p>
