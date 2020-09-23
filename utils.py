import cv2
import numpy as np


def plothand(img, kp, bones=None):
    """
     plothand is a function that plot single hand on the image
    If not specified, plothand use default bones to connect joints.
    Image is desired to read by opencv. The outputs of other form of inputs are not guaranteed

    Args:
        :param img : image in matrix representation
        :param kp : joints position in 2d (21,2)
        :param bones: specify the bone connections of joints. If None, use the default bones
    Returns:
        r1: sth. here....
        r2: sth. here....
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
        cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 3, colors[i, :], -1)
