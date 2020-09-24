import os

import cv2

from HUMBI.utils import *
from utils import hand_angle, plothand


def viz_sample(camera_list, inpath, outpath, save_fig=False, put_angle=False, is_left=True):
    """
    viz_sample is a function that visualize key points annotation of a single hand from this dataset

    Args:
        :param camera_list: camera calibration matrix
        :param inpath: input path of a particular frame, such as 'HUMBI/subject_1/hand/00000001'
        :param outpath: output path of the visualized image, if save_fig is True
        :param save_fig: whether to save the visualized image. default to False
        :param put_angle: whether to present hand angle info
        :param is_left: whether the hand is left hand. (1-left hand, 0-right hand, -1-unknown)
    Returns:
        :return: None
    """
    if is_left:
        indir = inpath + '/image_cropped/left/'
        kp_path = inpath + '/reconstruction/keypoints_l.txt'
    else:
        indir = inpath + '/image_cropped/right/'
        kp_path = inpath + '/reconstruction/keypoints_r.txt'
    kp_3d = read_kp_3d(kp_path)
    with open(indir + 'list.txt') as file:
        file_content = file.read()
        crop_info = file_content.split('\n')
    for info in crop_info:
        if info != '':
            id = int(info.split(' ')[0])
            img_name = 'image%07d.png' % id
            bbox = [float(num) for num in info.split(' ')[1:5]]

            scale_x = (bbox[1] - bbox[0] + 1) / 250
            scale_y = (bbox[3] - bbox[2] + 1) / 250
            # find corresponding camera matrix
            for cam in (item for item in camera_list if item['id'] == id):
                camera_param = cam
            M = camera_param['project']
            C = camera_param['C']
            R = camera_param['R']
            K = camera_param['intrinsic']
            T = - np.matmul(R, C)
            ex = np.hstack((R, T))  # 3*4
            M2 = np.matmul(K, ex)
            kp_2d = project2d(kp_3d, M2)  # 21*2

            kp_camera = project_camera(kp_3d, ex)  # 21*3
            angle = hand_angle(kp_camera)

            kp_2d[:, 0] = (kp_2d[:, 0] - bbox[0]) / scale_x
            kp_2d[:, 1] = (kp_2d[:, 1] - bbox[2]) / scale_y
            img = cv2.imread(indir + img_name)
            plothand(img, kp_2d)
            if put_angle:
                cv2.putText(img, 'angle: %.2f' % angle, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
            if save_fig:
                outpath_img = outpath + 'image%07d.jpg' % id
                cv2.imwrite(outpath_img, img)
            else:
                cv2.imshow(str(id), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def main():
    """
    this sample visualize all frames in subject_1 and frame 253 for both hand
    change dir and outpath to local path to the dataset and the output path
    """
    dir = 'D:/Hand-data/HUMBI/Hand_1_80_updat/'
    outpath = './sample'
    for sub in os.listdir(dir):
        if sub == 'subject_1':
            sub = dir + sub
            if os.path.isdir(sub):
                hand_dir = sub + '/hand/'
                camera_list = read_param(hand_dir)
                for d in os.listdir(hand_dir):
                    if d == '00000253':
                        in_dir = hand_dir + d
                        if os.path.isdir(in_dir):
                            viz_sample(camera_list, in_dir, outpath, is_left=True, put_angle=True)
                            viz_sample(camera_list, in_dir, outpath, is_left=False, put_angle=True)


if __name__ == '__main__':
    main()
