import cv2
import numpy as np
import os
import pandas as pd

class VisualOdometry:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.poses = self._load_poses(os.path.join(data_dir, '00.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'images'))
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.cur_pose = np.eye(4)

    @staticmethod
    def _load_calib(filepath):
        calib = pd.read_csv(filepath, delimiter=' ', header=None, index_col=0)
        P0 = np.array(calib.loc['P0:']).reshape((3, 4))
        K, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
        t1 = t1 / t1[3]
        return K, P0

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f:
                pose = np.fromstring(line, dtype=float, sep=' ').reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses.append(pose)
        return poses

    def _load_images(self, image_dir):
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    def process_frame(self, frame_id):
        img = self.images[frame_id]
        kp, des = self.orb.detectAndCompute(img, None)
        
        if self.prev_kp is not None:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, self.K)
            
            # Convert R and t to a 4x4 transformation matrix
            Rt = np.eye(4)
            Rt[:3, :3] = R
            Rt[:3, 3] = t.squeeze()
            
            # Update the current pose
            self.cur_pose = self.cur_pose @ Rt
        
        self.prev_kp = kp
        self.prev_des = des

    def get_current_pose(self):
        return self.cur_pose