import numpy as np


def read_param(inpath=''):
    """
     read camera calibration info for particular input path

     Args:
        :param inpath: the input subject path

    Returns:
        :return: camera calibration matrix, including intrinsic, extrinsic and projection
    """
    with open(inpath + 'intrinsic.txt') as file:
        file_content = file.read()
        list_in = file_content.split('\n')
    with open(inpath + 'extrinsic.txt') as file:
        file_content = file.read()
        list_ex = file_content.split('\n')
    with open(inpath + 'project.txt') as file:
        file_content = file.read()
        list_proj = file_content.split('\n')
    camera_list = []
    for i in range(3, len(list_in) - 4, 4):
        cameraid = list_in[i].split(' ')[1]
        ex_line = (i - 3) // 4 * 5 + 3
        C = [float(num) for num in list_ex[ex_line + 1].split(' ')]
        R = [[float(num) for num in list_ex[ex_line + j].split(' ')] for j in range(2, 5)]
        camera_extrinsic = [[float(num) for num in list_ex[ex_line + j].split(' ')] for j in range(1, 5)]  # 4*3
        camera_intrinsic = [[float(num) for num in list_in[i + j].split(' ')] for j in range(1, 4)]  # 3*3
        camera_proj = [[float(num) for num in list_proj[i + j].split(' ')] for j in range(1, 4)]  # 3*4
        camera_extrinsic = np.array(camera_extrinsic)
        camera_intrinsic = np.array(camera_intrinsic)
        camera_proj = np.array(camera_proj)
        R = np.array(R)
        C = np.array(C).reshape((3, 1))
        camera_dict = {'id': int(cameraid), 'intrinsic': camera_intrinsic, 'project': camera_proj,
                       'extrinsic': camera_extrinsic,
                       'C': C, 'R': R}
        camera_list.append(camera_dict)
    return camera_list


def read_kp_3d(path):
    with open(path) as file:
        file_content = file.read().split('\n')
        list = [[float(num) for num in file_content[i].split(' ')] for i in range(21)]
        kp = np.array(list)
    return kp


def project2d(point3d, camera):
    # point3d 21*3, camera 3*4, return 21*2
    ones = np.ones((1, 21))
    ap = np.vstack((point3d.T, ones))
    kp = np.matmul(camera, ap)  # 3*21
    kp = kp[0:2, :] / kp[2:, :]
    return kp.T


def project_camera(point3d, extr):
    # point3d 21*3 in world coord, extr: 3*4 extrinsic param
    ones = np.ones((1, 21))
    ap = np.vstack((point3d.T, ones))
    kp = np.matmul(extr, ap)  # 3*4 * 4*21 = 3*21
    return kp.T
