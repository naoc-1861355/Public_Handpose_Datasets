import json
import os

import cv2
import numpy as np
import scipy.io as sio

from utils import plothand

lr_trans = [120.054, 0, 0]
lr_k = [[822.79041, 0, 318.47345], [0, 822.79041, 250.31296], [0, 0, 1.]]

sk_trans = [-24.0381, -0.4563, -1.2326]
sk_rot = [[0.00531, -0.01196, 0.00301]]
sk_rot = cv2.Rodrigues(np.array(sk_rot))[0]
sk_color_k = [[607.92271, 0, 314.78337], [0, 607.88192, 236.42484], [0, 0, 1.]]
sk_depth_k = [[475.62768, 0, 336.41179], [0, 474.77709, 238.77962], [0, 0, 1.]]

out_path = './sample'


def trans_uvd(kp_3d):
    """
    perform camera projection from 3d key points annotation to uvd key points annotation

    :param kp_3d: 3d key points annotation in world coordination. It's in shape (3, 21)
    :return: uvd key points annotation in camera(2d) coordination. It's in shape (21,3)
    """
    uvd = np.mat(lr_k) * np.mat(kp_3d)
    uvd[:2, :] = uvd[:2, :] / uvd[2:, :]
    # uvd 3*21
    pts = uvd.T
    pts = np.array(pts)
    return pts


def viz(save_fig=False):
    """
    normdat is a function that convert this dataset to standard ezxr format output

    Args:
        :param save_fig: whether to save the visualized image. default to False

    Returns:
        :return: None
    """
    root = 'D:/Hand-data/STB/'
    out_dir = out_path
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

        lr_mat_data = sio.loadmat(os.path.join(labels_dir, img_dir + '_BB.mat'))['handPara']  # 3*21*1500
        sk_mat_data = sio.loadmat(os.path.join(labels_dir, img_dir + '_SK.mat'))['handPara']

        left_sample = left_imgs
        right_sample = right_imgs
        sk_sample = sk_color_imgs
        for i in range(len(left_imgs)):
            if i > 1:
                break
            # left
            left_img_path = os.path.join(root, img_dir, left_sample[i])
            ind_left = int((left_sample[i].strip('.png')).split('_')[-1])
            img = cv2.imread(left_img_path)
            label = lr_mat_data[..., ind_left]
            pts = trans_uvd(label)
            plothand(img, pts)
            if save_fig:
                if not os.path.exists(outpath):
                    os.mkdir(outpath)
                outpath_img = outpath + '/' + left_sample[i][:-4] + '.jpg'
                cv2.imwrite(outpath_img, img)
            else:
                cv2.imshow(left_sample[i][:-4], img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def main():
    viz()


if __name__ == '__main__':
    main()
