import cv2
import numpy as np


def plothand(img, kp, bones=None):
    """
     plothand is a function that plot single hand on the image

    Image is desired to read by opencv. The outputs of other form of inputs are not guaranteed

    Args:
        :param img : image in matrix representation
        :param kp : joints position in 2d (21,2)
        :param bones: specify the bone connections of joints. If None, use the default bones

    Returns:
        None
    """
    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    colors = colors[:, ::-1]
    colors = colors * 255
    # define connections and colors of the bones
    if bones is None:
        bones = [((0, 1), colors[1, :]),
                 ((1, 2), colors[2, :]),
                 ((2, 3), colors[3, :]),
                 ((3, 4), colors[4, :]),

                 ((0, 5), colors[5, :]),
                 ((5, 6), colors[6, :]),
                 ((6, 7), colors[7, :]),
                 ((7, 8), colors[8, :]),

                 ((0, 9), colors[9, :]),
                 ((9, 10), colors[10, :]),
                 ((10, 11), colors[11, :]),
                 ((11, 12), colors[12, :]),

                 ((0, 13), colors[13, :]),
                 ((13, 14), colors[14, :]),
                 ((14, 15), colors[15, :]),
                 ((15, 16), colors[16, :]),

                 ((0, 17), colors[17, :]),
                 ((17, 18), colors[18, :]),
                 ((18, 19), colors[19, :]),
                 ((19, 20), colors[20, :])]
    for connection, color in bones:
        kp1 = kp[connection[0]]
        kp2 = kp[connection[1]]
        cv2.line(img, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), color, 1, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 2, colors[i, :], -1)


def generate_json_2d(kps_2d, hand_bbox, is_left):
    """
    generate_json_2d is a function that combine single hand joint info into a ezxr format dict

    This function should only be used to generate 2d joint info.
    Args:
        :param hand_bbox : bounding box of joints
        :param kps_2d : joints info in 2d
        :param is_left : whether the hand is left

    Returns:
        :return dict_kp : joint info dictionary

    """
    pts = kps_2d
    f52 = {'x': pts[0, 0], 'y': pts[0, 1], 'd': -1}
    base_dic = {'x': -1, 'y': -1, 'z': -1}
    hand_3d_list = [base_dic for i in range(4)]
    f02 = [{'x': pts[0], 'y': pts[1], 'd': -1} for pts in pts[1:5, :]]
    f12 = [{'x': pts[0], 'y': pts[1], 'd': -1} for pts in pts[5:9, :]]
    f22 = [{'x': pts[0], 'y': pts[1], 'd': -1} for pts in pts[9:13, :]]
    f32 = [{'x': pts[0], 'y': pts[1], 'd': -1} for pts in pts[13:17, :]]
    f42 = [{'x': pts[0], 'y': pts[1], 'd': -1} for pts in pts[17:21, :]]
    dict3d = {'f%d3' % i: hand_3d_list for i in range(6)}
    dict_kp = {'palm_center': [-1, -1, -1], 'is_left': is_left, 'hand_bbox': hand_bbox, 'f52': [f52 for _ in range(4)],
               'f02': f02, 'f12': f12, 'f22': f22, 'f32': f32, 'f42': f42}
    dict_kp.update(dict3d)

    return dict_kp


def generate_json_3d(kps_2d, kps_3d, hand_bbox, is_left):
    """
    generate_json_3d is a function that combine single hand joint info into a ezxr format dict

    This function should only be used to generate 3d joint info.
    Args:
        :param kps_2d : joints info in 2d
        :param kps_3d: joints info in 3d
        :param hand_bbox : bounding box of joints
        :param is_left : whether the hand is left

    Returns:
        :return dict_kp : joint info dictionary

    """
    pts = kps_2d
    f52 = {'x': pts[0, 0], 'y': pts[0, 1], 'd': -1}
    f02 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[1:5, :]]
    f12 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[5:9, :]]
    f22 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[9:13, :]]
    f32 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[13:17, :]]
    f42 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[17:21, :]]
    xyz = kps_3d
    f53 = {'x': xyz[0, 0], 'y': xyz[0, 1], 'z': xyz[0, 2]}
    f03 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[1:5, :]]
    f13 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[5:9, :]]
    f23 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[9:13, :]]
    f33 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[13:17, :]]
    f43 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[17:21, :]]
    dict_kp = {'palm_center': [-1, -1, -1], 'is_left': is_left, 'hand_bbox': hand_bbox,
               'f52': [f52 for _ in range(4)], 'f53': [f53 for _ in range(4)],
               'f03': f03, 'f13': f13, 'f23': f23, 'f33': f33, 'f43': f43,
               'f02': f02, 'f12': f12, 'f22': f22, 'f32': f32, 'f42': f42
               }

    return dict_kp


def hand_angle(kpts):
    """
    hand_angle is a function that compute angle between hand and z-cord in camera coordination

    The hand plane is determined by index finger mcp, ring finger mcp and wrist point.
    Args:
        :param kpts : joints annotation in camera coordination in 3d (21, 3)

    Returns:
        :return: angle : angle in degree
    """
    p1 = kpts[0]
    p2 = kpts[5]
    p3 = kpts[13]
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    coord = np.array([0, 0, 1])
    cos = np.dot(coord, cp) / (np.linalg.norm(coord) * np.linalg.norm(cp))
    angle = np.rad2deg(np.arccos(abs(cos)))
    return angle
