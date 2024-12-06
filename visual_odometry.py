import cv2
import numpy as np
import os
import pandas as pd

class VisualOdometry:
    '''
    Class handling the visual odometry pipeline. Called from the main file, processes the kitti images and gives us back the current pose of the camera.
    '''
    def __init__(self, data_dir):
        self.data_dir = data_dir # image directory
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt')) # load camera calibration matrix and projection matrix from file
        self.poses = self._load_poses(os.path.join(data_dir, '00.txt')) # Load ground truth poses
        self.images = self._load_images(os.path.join(data_dir, 'images')) # Load in the images
        self.orb = cv2.ORB_create() # Initialize the ORB featured detector and descriptor
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Initialize the brute-force matcher
        self.prev_kp = None # Var for keeping track of the previous keypoints
        self.prev_des = None # Var for keeping track of the previous descriptors
        self.cur_pose = np.eye(4) # Initialize the current pose to the identity matrix

    @staticmethod # Method to handle the loading of the calibration file
    def _load_calib(filepath):
        calib = pd.read_csv(filepath, delimiter=' ', header=None, index_col=0) # Read data from file
        P0 = np.array(calib.loc['P0:']).reshape((3, 4)) # Extract the projection matrix P0 as this is the camera we are working with
        K, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P0) # Decompose projection into intrinsic K, rotation matrix, and the translation vector
        t1 = t1 / t1[3] # Translation vector normalization
        return K, P0 # Return the intrinsic matrix and the projection matrix

    @staticmethod # Method for loading in the ground truth poses
    def _load_poses(filepath):
        poses = [] # List to store the poses
        with open(filepath, 'r') as f: # Open file and read in each line
            for line in f: 
                pose = np.fromstring(line, dtype=float, sep=' ').reshape(3, 4) # Read in as np array and reshape to 3x4
                pose = np.vstack((pose, [0, 0, 0, 1])) # Add in the 0001 vector to make it a 4x4 transformation matrix
                poses.append(pose) # Append that line to the list
        return poses # Return the poses

    def _load_images(self, image_dir): # Function for loading in the images
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]) # Sorted list from the image directory
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths] # Load in each image as grayscale, images are already grayscale - this is just to make sure
        return images # Return the images

    def process_frame(self, frame_id): # Function for the VO processing we do over each frame
        img = self.images[frame_id] # Get the image ID corresponding to the current frame
        kp, des = self.orb.detectAndCompute(img, None) # Use ORB to detect the keypoints and compute descriptor vectors
        
        if self.prev_kp is not None: # If we have keypoints, proceed
            matches = self.bf.match(self.prev_des, des) # Match descriptors between frames
            matches = sorted(matches, key=lambda x: x.distance) # Sort the matches by distance
            src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) # Get the source points
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) # Get the destination points
            E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0) # Compute the essential matrix using RANSAC to handle any keypoint outliers
            _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, self.K) # Recover the pose (translation and rotation) from the computed essential matrix
            
            Rt = np.eye(4) # Convert the computed rotation and translation into the 4x4 transformation matrix representing the camera position in the world frame
            Rt[:3, :3] = R
            Rt[:3, 3] = t.squeeze()
            
            # Update the pose with the computed transformation matrix
            self.cur_pose = self.cur_pose @ Rt
        
        # Upate the previous keypoints and descriptors with the current computed ones
        self.prev_kp = kp
        self.prev_des = des

    # Finally we return the current pose
    def get_current_pose(self):
        return self.cur_pose