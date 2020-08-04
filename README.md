# Computer Vision Projects - Classical and Modern Approaches
Contains Computer Vision projects I implemented out of interest. 

Projects aimed at:
1. Classical Computer Vision Techniques
2. Advanced Computer Vision Techniques
3. Computer Vision using ML/DL

# Classical Computer Vision Projects

## Virtual Pen and Eraser  
- [x] Completed

I recently came across posts on linkedin, where people were drawing virtually on the screen, originally posted by Mr. Taha Anwar. So, I thought why not try it on my own! Though the original version works on color thresholding to extract the pen from the scene, I had a hard time(& changing light intensities in my background) finding a right range. Hence, I added an aruco marker(tag id: 0) to act as a pen. Feel free to use this version and build your version upon it. A couple of other id markers can work for various colors and even eraser.
<p align="center">
<img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/VirtualPen%26Eraser/GIF_VirtualPen_2_Small.gif" alt="OK?OK" width="250"/>
</p>

## Panaroma Creator  
- [x] Completed

In this mini-mini-project, I have build a custom panaroma script using Feature Matching through ORB algorithm. This is followed by Image Alignment after finding the Homogenous Transformation between the two images and applying perspective transformation to one of the image.
<p align="center">
<img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/CreatePanaroma/Result.jpg" alt="PanoramaResult" width="250"/>
</p>

## [Buoy Detection using Gaussian Mixture Models](https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/tree/master/BuoyDetectionGMM)

- [x] Completed

This project introduced the concept of color segmentation using Gaussian Mixture Models and Expectation Maximization techniques. The video sequence provided has been captured underwater and shows three buoys of different colors, namely yellow, orange and green. They are almost circular in shape and are distinctly colored. However, conventional segmentation techniques involving color thresholding will not work well in such an environment, since noise and varying light intensities will render any hard-coded thresholds ineffective. In such a scenario, we “learn” the color distributions of the buoys and use that learned model to segment them. This project required us to obtain a tight segmentation of each buoy for the entire video sequence by applying a tight contour (in the respective color of the buoy being segmented) around each buoy.

<p align="center">
  <img src="https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/blob/master/BuoyDetectionGMM/buoyDetection.gif" alt="Underwater Buoy Detection using Gaussian Mixture Models(GMMs)"  width="250"/>
</p>

## [Lane Detection using Sliding Windows](https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/tree/master/AdvanceLaneDetection)
 
- [x] Completed

In this project we aim to do simple Lane Detection to mimic Lane Departure Warning systems used in SelfDriving Cars.  We are provided with two video sequences(one in normal lighting and other one in changing light intensities), taken from a self driving car.  Our task was to design an algorithm to detect lanes on the road,as well as estimate the road curvature to predict car turns.

<p align="center">
  <img src="https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/blob/master/AdvanceLaneDetection/laneDetection.gif" alt="Advance Lane Detection for Autonomous Cars"  width="250"/>
</p>

# Advanced Computer Vision Techniques

## Structure from Motion  
- [x] Completed

In this project, I performed structure from motion(a.k.a structure and motion) on an image dataset. The project was extensive and it required a good hand in geometrical transformation techniques. Overall, I am happy(somewhat) to finish it but there are a great amount of modifications I see possible to make it better. I will update this project and make it better soon. Here is my course of action for now:
<p align="center">

- [x] Camera Calibration for custom dataset.  
- [x] Feature Tracking using Optic Flow.  
- [x] Image Rectification.  
- [x] Triangulation.  
- [x] 3D Point Cloud Generation.  
</p>
<p align="center">
<img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/StructureFromMotion/Front.png" alt="SFM" width="250"/>
<img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/StructureFromMotion/SideView.png" alt="SFM" width="250"/>
</p>

## [Visual Odometry](https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/tree/master/VisualOdometry)
- [x] Completed

Visual Odometry is a crucial concept in Robotics Perception for estimating the trajectory of the robot (the camera on the robot to be precise). The concepts involved in Visual Odometry are quite the same for SLAM which needless to say is an integral part of Perception.

In this project we have frames of a driving sequence taken by a camera in a car, and the scripts to extract the intrinsic parameters. The concepts for Visual Odometry and Structure from Motion are mostly same. After Getting the Essential Matrix, I retrieved Camera's rotation and translation to plot its trajectory.

<p align="center">
<img src="https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/blob/master/VisualOdometry/visualodom.gif" alt="VO" width="250"/>
</p>

## Augmented Reality  
- [ ] Completed

In this project, I am placing multiple 3d objects(solid ones) on a flat surface. Coming Soon!   . . .

# Computer Vision through Deep Learning Projects

## Facial Keypoints Recognition  
- [x] Completed

In this project, I combined my knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face! Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications.

The project is broken down into four main parts:

[Notebook 1](FacialKeypointRecognition/1_LoadAndVisualizeData.ipynb): Loading and Visualizing the Facial Keypoint Data  
[Notebook 2](FacialKeypointRecognition/2_Define_TheNetworkArchitecture.ipynb): Defining and Training a Convolutional Neural Network (CNN) to predict Facial Keypoints  
[Notebook 3](FacialKeypointRecognition/3_FacialKeypointDetectionCompletePipeline.ipynb): Facial Keypoint Detection Using Haar Cascades and our trained CNN.  
[Notebook 4](FacialKeypointRecognition/4_ApplicationsKeypoints.ipynb): Applications using Facial Keypoints


## [Traffic Sign Detection](https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/tree/master/TrafficSignsDetection)  
- [x] Completed

The focus of this project is divided into two pipelines, Traffic sign detection and traffic sign classification. Traffic sign detection is the process of forming a bounding box around a traffic sign, so that the region of interest can be cropped and given to the classifier for sign classification. HSV feature and MSER features are tested for robust sign detection. Model is trained using Support Vector Machines.

<p align="center">
  <img src="https://github.com/SarveshRobotics/Perception-For-Autonomous-Robots/blob/master/TrafficSignsDetection/GIF_TrafficSignDetect.gif" alt="Homography and Pose Estimation" width="250"/>
</p>

## MNIST Digit Recognition - SVM with HOG
- [x] Completed

In this project, I trained SVM classifier(with RBF kernel) using HOG descriptors for a final accuracy of 99.1% on the MNIST digit dataset. I plan to use this model to make a virtual calculator later on.

<p align="center">
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/MNIST%20Digit%20Recognition/DigitRecognition.png" alt="MNIST Digits Test" width="600"/>
</p>

## Image Restoration and Colorization
- [x] Completed

I had an old Black&White picture of my parents back from 80s and now it has got cracks on it as well. This gave me an idea to create a restoration and colorization pipeline. For the restoration I used, cv2.TELEA method, which basically approximates the neighbourhood pixels for the white pixels in the mask. Mask had to be created manually for this problem, though for digital images, it can be created automatically. Later, I passed the restored image to a CNN network to colorize the image. For the colorization, I used the network developed in this [paper](ImageInpainting_ImageColorization/Colorization.pdf) by Zhang et. al. This model uniquely converts the colorization task to be a classification task. To reduce the number of parameters to be distinguished, model works on LAB format image. L channel is equivalent to the gray channel intensities and the task remains to distinguish between A and B channel. An intelligent approach has been adapted in the paper and A and B channel are combined, so this is reduced to a multiclass classification problem for each pixel. I tried the model on a badly taken picture(and edited too with stamp) from mobile camera back in 2006s of my Father's black&white photograph. Here is the result:

<p align="center">
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/ImageInpainting_ImageColorization/Papa.jpg" alt="Original" width="200"/>
  
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/ImageInpainting_ImageColorization/maskPapaNew.jpg" alt="Mask" width="200"/>
  
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/ImageInpainting_ImageColorization/RestoredImage.jpg" alt="RestoredAndColorizedImage" width="200"/>
</p>

## Neural Style Transfer
- [x] Completed

I recently came across a new concept in Computer Vision, Neural Style Transfer. I tried a trained network in PyTorch and the results are impressive. To get deep into it, I am creating a custom pipeline, will be hosting its results soon. Here is the pre-trained network's results:

<p align="center">
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/NeuralStyleTransfer/testImages/AchiPhoto.jpg" alt="Original" width="200"/>
  
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/NeuralStyleTransfer/testImages/Outputfeathers.jpg" alt="Mask" width="200"/>
  
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/NeuralStyleTransfer/testImages/Outputmosaic.jpg" alt="RestoredAndColorizedImage" width="200"/>
</p>

## Image Similarity  
- [x] Completed

Today, I finished a very interesting project based on Image Similarity. Basically, it returns user all(custom) of the images it found similar to the query image. I learned a lot of new things while completing this. I came to know about FAST.AI that is making life easier for DL practitioners. I used Fast.AI to accomplish transfer learning from RESNET34 CNN. RESNET34 is originally trained on CIFAR-10 dataset. I initially retrained the last layer of the model on CALTECH101 dataset and then retrained all previous layers to get accuracy of 94.6%. Next, I created Forward Hook to store embeddings of all the images in CALTECH101 dataset(9000+ images). Using Locality Sensitivity Hashing, I wrote a small script to return the user similar images it found to the query image. Here is the result:

<p align="center">
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/ImageSimilarity/Query.png" alt="Query" width="200"/>
  
  <img src="https://github.com/SarveshRobotics/ComputerVisionProjects/blob/master/ImageSimilarity/Results.png" alt="Results" width="400"/>
</p>



