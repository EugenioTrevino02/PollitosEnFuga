import os
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

class CameraPoses:
    def __init__(self, intrinsic):
        self.K = intrinsic
        self.extrinsic = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0]])
        self.P = self.K @ self.extrinsic
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.world_points = []
        self.current_pose = None

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_world_points(self):
        return np.array(self.world_points)
    
    def get_matches(self, img1, img2):
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if len(kp1) > 6 and len(kp2) > 6 and des1 is not None and des2 is not None:
            matches = self.flann.knnMatch(des1, des2, k=2)
            good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.5 * n.distance]
            if len(good_matches) > 0:
                q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                return q1, q2
            else:
                return None, None
        else:
            return None, None

    def get_pose(self, q1, q2):
        E, mask = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(E, q1, q2)
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1, np.ndarray.flatten(t))
        T2 = self._form_transf(R2, np.ndarray.flatten(t))
        T3 = self._form_transf(R1, np.ndarray.flatten(-t))
        T4 = self._form_transf(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max_idx = np.argmax(positives)
        best_pair = [R1, -t] if max_idx == 2 else [R2, -t] if max_idx == 3 else [R1, t] if max_idx == 0 else [R2, t]
        return best_pair[0], np.ndarray.flatten(best_pair[1]) * np.mean(positives)

def load_intrinsic_parameters(filepath):
    return np.load(filepath)

def main(video_path, intrinsic_path):
    intrinsic = load_intrinsic_parameters(intrinsic_path)
    camera = CameraPoses(intrinsic)
    
    cap = cv2.VideoCapture(video_path)
    
    camera_poses = []
    current_pose = np.eye(4)
    
    prev_img = None
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        start_time = time.perf_counter()
        
        if prev_gray is not None:
            q1, q2 = camera.get_matches(prev_gray, gray)
            if q1 is not None and q2 is not None:
                transformation_matrix = camera.get_pose(q1, q2)
                current_pose = current_pose @ transformation_matrix
                camera_poses.append(current_pose)
        
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i in range(3):
            for j in range(3):
                cv2.putText(frame, str(np.round(current_pose[i, j], 2)), (260 + 80 * j, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        
        for i in range(3):
            cv2.putText(frame, str(np.round(current_pose[i, 3], 2)), (540, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = gray
    
    cap.release()
    cv2.destroyAllWindows()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    trajectory = np.array([pose[:3, 3] for pose in camera_poses])
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Camera Path')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    video_path = 'patosVid.mp4'  # Replace with your video file path
    intrinsic_path = 'intrinsicNew.npy'  # Replace with your intrinsic parameters file path
    main(video_path, intrinsic_path)
