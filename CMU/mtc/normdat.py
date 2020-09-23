import json
import os
import pickle

import cv2
import numpy as np

from utils import hand_angle, plothand, generate_json_3d

hand_connMat = np.array(
    [0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 0, 9, 9, 10, 10, 11, 11, 12, 0, 13, 13, 14, 14, 15, 15, 16, 0, 17,
     17, 18, 18, 19, 19, 20]).reshape(-1, 2)

# don't include chinese character in path, or cv2 read image will fail
folder_path = '\mtc_dataset\hand_data'


def project(joints, calib, apply_distort=True):
    """
    project is a function that project 3d world joints to 2d and 3d joints in camera coordination

    Args:
        :param joints: joints in 3d world coordination. N * 3 numpy array.
        :param calib: a dict containing 'R', 'K', 't', 'distCoef' (numpy array)

    Returns:
        :return pt: N * 2 numpy array
        :return pt_cam: N * 3 numpy array in camera coordination
    """
    x = np.dot(calib['R'], joints.T) + calib['t']
    xp = x[:2, :] / x[2, :]

    if apply_distort:
        x2 = xp[0, :] * xp[0, :]
        y2 = xp[1, :] * xp[1, :]
        xy = x2 * y2
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2

        dc = calib['distCoef']
        radial = 1.0 + dc[0] * r2 + dc[1] * r4 + dc[4] * r6
        tan_x = 2.0 * dc[2] * xy + dc[3] * (r2 + 2.0 * x2)
        tan_y = 2.0 * dc[3] * xy + dc[2] * (r2 + 2.0 * y2)

        # xp = [radial;radial].*xp(1:2,:) + [tangential_x; tangential_y]
        xp[0, :] = radial * xp[0, :] + tan_x
        xp[1, :] = radial * xp[1, :] + tan_y

    # pt = bsxfun(@plus, cam.K(1:2,1:2)*xp, cam.K(1:2,3))';
    pt = np.dot(calib['K'][:2, :2], xp) + calib['K'][:2, 2].reshape((2, 1))

    return pt.T, x.T


def normdat(outpath):
    """
     normdat is a function that crop and visualize keypoint annotation on image

     Args:
         :param outpath: the output path of image and json file

     Returns:
        :return None
    """
    with open(os.path.join(folder_path + 'annotation.pkl'), 'rb') as f:
        data = pickle.load(f)

    with open(os.path.join(folder_path + 'camera_data.pkl'), 'rb') as f:
        cam = pickle.load(f)

    mode_data = data['training_data']
    # mode_data = data['testing_data']

    for i, sample in enumerate(mode_data):
        if i > 15:
            break
        seq_name = sample['seqName']
        frame_str = sample['frame_str']
        frame_path = os.path.join(folder_path, 'hdImgs/{}/{}'.format(seq_name, frame_str))
        outdir_parent = os.path.join(outpath, seq_name, frame_str)

        if 'left_hand' in sample:
            # this means left hands exists, then we project 3d to 31 different camera
            outdir = os.path.join(outdir_parent, 'left')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for c in range(31):
                img_name = os.path.join(frame_path, '00_{:02d}_{}.jpg'.format(c, frame_str))
                # since some camera image is missing, we need to judge whether image exists
                if os.path.exists(img_name) and sum(sample['left_hand']['2D'][c]['occluded']) <= 5:
                    calib_data = cam[seq_name][c]
                    # kp process
                    left_hand_landmark = np.array(sample['left_hand']['landmarks']).reshape(-1, 3)
                    left_kp_2d, camera_3d = project(left_hand_landmark, calib_data)
                    angle = hand_angle(camera_3d)
                    if angle < 50:
                        # image process
                        x_min = min(left_kp_2d[:, 0])
                        x_max = max(left_kp_2d[:, 0])
                        y_min = min(left_kp_2d[:, 1])
                        y_max = max(left_kp_2d[:, 1])
                        hand_bbox = [x_min, x_max, y_min, y_max]
                        dict_kp = generate_json_3d(left_kp_2d, camera_3d, hand_bbox, 0)
                        outpath_img = os.path.join(outdir, '00_{:02d}_{}.jpg'.format(c, frame_str))
                        outpath_json = os.path.join(outdir, '00_{:02d}_{}.json'.format(c, frame_str))
                        img = cv2.imread(img_name)
                        cv2.imwrite(outpath_img, img)
                        with open(outpath_json, 'w') as outfile:
                            json.dump(dict_kp, outfile)
        if 'right_hand' in sample:
            # this means left hands exists, then we project 3d to 31 different camera
            outdir = os.path.join(outdir_parent, 'right')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for c in range(31):
                img_name = os.path.join(frame_path, '00_{:02d}_{}.jpg'.format(c, frame_str))
                # since some camera image is missing, we need to judge whether image exists
                if os.path.exists(img_name) and sum(sample['right_hand']['2D'][c]['occluded']) <= 5:
                    calib_data = cam[seq_name][c]
                    # kp process
                    right_hand_landmark = np.array(sample['right_hand']['landmarks']).reshape(-1, 3)
                    right_kp_2d, camera_3d = project(right_hand_landmark, calib_data)
                    angle = hand_angle(camera_3d)
                    if angle < 50:
                        # image process
                        x_min = min(right_kp_2d[:, 0])
                        x_max = max(right_kp_2d[:, 0])
                        y_min = min(right_kp_2d[:, 1])
                        y_max = max(right_kp_2d[:, 1])
                        hand_bbox = [x_min, x_max, y_min, y_max]
                        dict_kp = generate_json_3d(right_kp_2d, camera_3d, hand_bbox, 0)
                        outpath_img = os.path.join(outdir, '00_{:02d}_{}.jpg'.format(c, frame_str))
                        outpath_json = os.path.join(outdir, '00_{:02d}_{}.json'.format(c, frame_str))
                        img = cv2.imread(img_name)
                        cv2.imwrite(outpath_img, img)
                        with open(outpath_json, 'w') as outfile:
                            json.dump(dict_kp, outfile)


def main():
    outpath = 'sample'
    normdat(outpath)


if __name__ == '__main__':
    main()
