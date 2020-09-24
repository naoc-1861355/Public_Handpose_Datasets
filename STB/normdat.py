import json
import os

import cv2
import numpy as np
import scipy.io as sio

lr_trans = [120.054, 0, 0]
lr_k = [[822.79041, 0, 318.47345], [0, 822.79041, 250.31296], [0, 0, 1.]]

sk_trans = [-24.0381, -0.4563, -1.2326]
sk_rot = [[0.00531, -0.01196, 0.00301]]
sk_rot = cv2.Rodrigues(np.array(sk_rot))[0]
sk_color_k = [[607.92271, 0, 314.78337], [0, 607.88192, 236.42484], [0, 0, 1.]]
sk_depth_k = [[475.62768, 0, 336.41179], [0, 474.77709, 238.77962], [0, 0, 1.]]


def trans_uvd(kp_3d):
    """
    perform camera projection from 3d key points annotation to uvd key points annotation

    :param kp_3d: 3d keypoints annotation in world coordination. It's in shape (3, 21)
    :return: uvd keypoints annotation in camera(2d) coordination. It's in shape (21,3)
    """
    uvd = np.mat(lr_k) * np.mat(kp_3d)
    uvd[:2, :] = uvd[:2, :] / uvd[2:, :]
    # uvd 3*21
    pts = uvd.T
    pts = np.array(pts)
    return pts


def generate_json_3d(kp_uvd, is_left):
    """
    generate_json_3d is a function that combine single hand joint info into a ezxr format dict

    This function should only be used in STB data, which contains palm center instead of wrist point
    Args:
        :param kps_2d : joints info in 2d
        :param kps_3d: joints info in 3d
        :param hand_bbox : bounding box of joints
        :param is_left : whether the hand is left

    Returns:
        :return dict_kp : joint info dictionary

    """
    pts = kp_uvd
    x_max = max(pts[:, 0])
    x_min = min(pts[:, 0])
    y_max = max(pts[:, 1])
    y_min = min(pts[:, 1])
    hand_bbox = [x_min, x_max, y_min, y_max]
    f52 = {'x': -1, 'y': -1, 'd': -1}
    f42 = [{'x': pt[0, 0], 'y': pt[0, 1], 'd': pt[0, 2]} for pt in pts[1:5, :]]
    f32 = [{'x': pt[0, 0], 'y': pt[0, 1], 'd': pt[0, 2]} for pt in pts[5:9, :]]
    f22 = [{'x': pt[0, 0], 'y': pt[0, 1], 'd': pt[0, 2]} for pt in pts[9:13, :]]
    f12 = [{'x': pt[0, 0], 'y': pt[0, 1], 'd': pt[0, 2]} for pt in pts[13:17, :]]
    f02 = [{'x': pt[0, 0], 'y': pt[0, 1], 'd': pt[0, 2]} for pt in pts[17:21, :]]

    xyz = kp_uvd
    f53 = {'x': -1, 'y': -1, 'z': -1}
    f43 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[1:5, :]]
    f33 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[5:9, :]]
    f23 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[9:13, :]]
    f13 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[13:17, :]]
    f03 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[17:21, :]]
    dict_kp = {'palm_center': [pts[0, 0], pts[0, 1], pts[0, 2]], 'is_left': is_left, 'hand_bbox': hand_bbox,
               'f52': [f52 for _ in range(4)], 'f53': [f53 for _ in range(4)],
               'f03': f03, 'f13': f13, 'f23': f23, 'f33': f33, 'f43': f43,
               'f02': f02, 'f12': f12, 'f22': f22, 'f32': f32, 'f42': f42}
    return dict_kp


def normdat(out_dir):
    """
    normdat is a function that convert this dataset to standard ezxr format output

    Args:
        :param out_dir: the output path of the formatted dataset

    Returns:
        :return: None
    """
    # root = '//10.244.10.33/disk4/public_handpose_dataset/STB/'
    root = 'D:/Hand-data/STB/'
    dirs = os.listdir(root)
    dirs = [i for i in dirs if i.startswith('B6') and os.path.isdir(os.path.join(root, i))]
    labels_dir = os.path.join(root, 'labels')

    # exclude_dir = ['B1Counting', 'B1Random', 'B2Counting', 'B2Random']
    # dirs = [x for x in dirs if x not in exclude_dir]
    for img_dir in dirs:
        imgs_name = os.listdir(os.path.join(root, img_dir))
        imgs_name = [i for i in imgs_name if i.endswith('.png')]
        left_imgs = [i for i in imgs_name if 'BB_left' in i]
        right_imgs = [i for i in imgs_name if 'BB_right' in i]
        sk_color_imgs = [i for i in imgs_name if 'SK_color' in i]
        outpath = os.path.join(out_dir, img_dir)
        if not os.path.exists(outpath):
            os.mkdir(outpath)

        lr_mat_data = sio.loadmat(os.path.join(labels_dir, img_dir + '_BB.mat'))['handPara']  # 3*21*1500
        sk_mat_data = sio.loadmat(os.path.join(labels_dir, img_dir + '_SK.mat'))['handPara']

        left_sample = left_imgs
        right_sample = right_imgs
        sk_sample = sk_color_imgs
        for i in range(len(left_imgs)):
            # left
            left_img_path = os.path.join(root, img_dir, left_sample[i])
            ind_left = int((left_sample[i].strip('.png')).split('_')[-1])
            img = cv2.imread(left_img_path)
            label = lr_mat_data[..., ind_left]
            pts = trans_uvd(label)
            dict_kp = generate_json_3d(pts, is_left=1)
            outpath_img = outpath + '/' + left_sample[i][:-4] + '.jpg'
            outpath_json = outpath + '/' + left_sample[i][:-4] + '.json'
            cv2.imwrite(outpath_img, img)
            # print(dict_kp)
            with open(outpath_json, 'w') as outfile:
                json.dump(dict_kp, outfile)

            # right
            right_img_path = os.path.join(root, img_dir, right_sample[i])
            ind_right = int((right_sample[i].strip('.png')).split('_')[-1])
            img = cv2.imread(right_img_path)
            right_label = lr_mat_data[..., ind_right]
            label = right_label - np.mat(lr_trans).reshape([3, 1])
            pts = trans_uvd(label)
            dict_kp = generate_json_3d(pts, is_left=0)
            outpath_img = outpath + '/' + right_sample[i][:-4] + '.jpg'
            outpath_json = outpath + '/' + right_sample[i][:-4] + '.json'
            cv2.imwrite(outpath_img, img)
            with open(outpath_json, 'w') as outfile:
                json.dump(dict_kp, outfile)

            # sk
            sk_img_path = os.path.join(root, img_dir, sk_sample[i])
            ind_sk = int((sk_sample[i].strip('.png')).split('_')[-1])
            img = cv2.imread(sk_img_path)
            sk_label = sk_mat_data[..., ind_sk]

            label = np.mat(sk_rot).transpose() * (sk_label - np.mat(sk_trans).reshape([3, 1]))
            pts = trans_uvd(label)
            dict_kp = generate_json_3d(pts, is_left=-1)
            outpath_img = outpath + '/' + sk_sample[i][:-4] + '.jpg'
            outpath_json = outpath + '/' + sk_sample[i][:-4] + '.json'
            cv2.imwrite(outpath_img, img)
            with open(outpath_json, 'w') as outfile:
                json.dump(dict_kp, outfile)


def main():
    out_dir = 'data/'
    normdat(out_dir)


if __name__ == '__main__':
    main()
