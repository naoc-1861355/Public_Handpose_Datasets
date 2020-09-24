import fnmatch
import os
import pickle
import re

import cv2
import numpy as np
from utils import plothand

def saveAnnotation(jointCamPath, positions):
    fOut = open(jointCamPath, 'w')
    fOut.write("F4_KNU1_A " + str(positions[0][0]) + " " + str(positions[0][1]) + "\n")
    fOut.write("F4_KNU1_B " + str(positions[1][0]) + " " + str(positions[1][1]) + "\n")
    fOut.write("F4_KNU2_A " + str(positions[2][0]) + " " + str(positions[2][1]) + "\n")
    fOut.write("F4_KNU3_A " + str(positions[3][0]) + " " + str(positions[3][1]) + "\n")

    fOut.write("F3_KNU1_A " + str(positions[4][0]) + " " + str(positions[4][1]) + "\n")
    fOut.write("F3_KNU1_B " + str(positions[5][0]) + " " + str(positions[5][1]) + "\n")
    fOut.write("F3_KNU2_A " + str(positions[6][0]) + " " + str(positions[6][1]) + "\n")
    fOut.write("F3_KNU3_A " + str(positions[7][0]) + " " + str(positions[7][1]) + "\n")

    fOut.write("F1_KNU1_A " + str(positions[8][0]) + " " + str(positions[8][1]) + "\n")
    fOut.write("F1_KNU1_B " + str(positions[9][0]) + " " + str(positions[9][1]) + "\n")
    fOut.write("F1_KNU2_A " + str(positions[10][0]) + " " + str(positions[10][1]) + "\n")
    fOut.write("F1_KNU3_A " + str(positions[11][0]) + " " + str(positions[11][1]) + "\n")

    fOut.write("F2_KNU1_A " + str(positions[12][0]) + " " + str(positions[12][1]) + "\n")
    fOut.write("F2_KNU1_B " + str(positions[13][0]) + " " + str(positions[13][1]) + "\n")
    fOut.write("F2_KNU2_A " + str(positions[14][0]) + " " + str(positions[14][1]) + "\n")
    fOut.write("F2_KNU3_A " + str(positions[15][0]) + " " + str(positions[15][1]) + "\n")

    fOut.write("TH_KNU1_A " + str(positions[16][0]) + " " + str(positions[16][1]) + "\n")
    fOut.write("TH_KNU1_B " + str(positions[17][0]) + " " + str(positions[17][1]) + "\n")
    fOut.write("TH_KNU2_A " + str(positions[18][0]) + " " + str(positions[18][1]) + "\n")
    fOut.write("TH_KNU3_A " + str(positions[19][0]) + " " + str(positions[19][1]) + "\n")
    fOut.write("PALM_POSITION " + str(positions[20][0]) + " " + str(positions[20][1]) + "\n")
    fOut.close()


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches


def readAnnotation3D(file):
    f = open(file, "r")
    an = []
    for l in f:
        l = l.split()
        an.append((float(l[1]), float(l[2]), float(l[3])))

    return np.array(an, dtype=float)


def getCameraMatrix():
    Fx = 614.878
    Fy = 615.479
    Cx = 313.219
    Cy = 231.288
    cameraMatrix = np.array([[Fx, 0, Cx],
                             [0, Fy, Cy],
                             [0, 0, 1]])
    return cameraMatrix


def getDistCoeffs():
    return np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])


def viz(inpath, outpath, save_fig=False):
    """
    viz_sample is a function that visualize key points annotation of a single hand from this dataset

    Args:
        :param inpath: path to this dataset
        :param outpath: output path of the visualized image, if save_fig is True
        :param save_fig: whether to save the visualized image. Default to False

    :return: None
    """
    pathToDataset = inpath

    cameraMatrix = getCameraMatrix()
    distCoeffs = getDistCoeffs()

    outputdir = outpath
    # iterate sequences
    for i in os.listdir(pathToDataset):
        # read the color frames
        path = pathToDataset + i + "/"
        colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
        colorFrames = natural_sort(colorFrames)
        print("There are", len(colorFrames), "color frames on the sequence data_" + str(i))
        # read the calibrations for each camera
        print("Loading calibration for ../calibrations/" + i)
        # f = open("../calibrations/data_" + str(i) + "/webcam_1/rvec.pkl", "r")
        c_0_0 = pickle.load(open("../calibrations/" + i + "/webcam_1/rvec.pkl", "r"))
        c_0_1 = pickle.load(open("../calibrations/" + i + "/webcam_1/tvec.pkl", "r"))
        c_1_0 = pickle.load(open("../calibrations/" + i + "/webcam_2/rvec.pkl", "r"))
        c_1_1 = pickle.load(open("../calibrations/" + i + "/webcam_2/tvec.pkl", "r"))
        c_2_0 = pickle.load(open("../calibrations/" + i + "/webcam_3/rvec.pkl", "r"))
        c_2_1 = pickle.load(open("../calibrations/" + i + "/webcam_3/tvec.pkl", "r"))
        c_3_0 = pickle.load(open("../calibrations/" + i + "/webcam_4/rvec.pkl", "r"))
        c_3_1 = pickle.load(open("../calibrations/" + i + "/webcam_4/tvec.pkl", "r"))

        for j in range(len(colorFrames)):
            toks1 = colorFrames[j].split("/")
            toks2 = toks1[3].split("_")
            jointPath = toks1[0] + "/" + toks1[1] + "/" + toks1[2] + "/" + toks2[0] + "_joints.txt"
            points3d = readAnnotation3D(jointPath)[0:21]  # the last point is the normal

            # project 3d LM points to the image plane
            webcam_id = int(toks2[2].split(".")[0]) - 1
            # print("Calibration for webcam id:",webcam_id)
            if webcam_id == 0:
                rvec = c_0_0
                tvec = c_0_1
            elif webcam_id == 1:
                rvec = c_1_0
                tvec = c_1_1
            elif webcam_id == 2:
                rvec = c_2_0
                tvec = c_2_1
            elif webcam_id == 3:
                rvec = c_3_0
                tvec = c_3_1

            pts2d, _ = cv2.projectPoints(points3d, rvec, tvec, cameraMatrix, distCoeffs)

            edges = [[20, 1], [1, 0], [0, 2], [2, 3], [20, 5], [5, 4], [4, 6], [6, 7], [20, 9], [9, 8], [8, 10],
                     [10, 11], [20, 13],
                     [13, 12], [12, 14], [14, 15], [20, 17], [17, 16], [16, 18], [18, 19]]
            img = cv2.imread(colorFrames[j])
            for k in range(pts2d.shape[0]):
                cv2.circle(img, (int(pts2d[k][0][0]), int(pts2d[k][0][1])), 3, (0, 0, 255), -1)
            # cv2.putText(img, str(k), (int(points2d[k][0][0]), int(points2d[k][0][1])), font, 0.3, (0, 0, 255), 2)
            for edge in edges:
                p1 = pts2d[edge[0]]
                p2 = pts2d[edge[1]]
                cv2.line(img, (int(p1[0][0]), int(p1[0][1])), (int(p2[0][0]), int(p2[0][1])), (0, 255, 0), 1,
                         cv2.LINE_AA)
            if save_fig:
                cv2.imwrite(outpath+'%d.jpg' % j, img)
            else:
                cv2.imshow(str(j), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

def main():
    pathToDataset = "../annotated_frames/"
    outpath = "../out_2"
    viz(pathToDataset, outpath)


if __name__ == "__main__":
    main()
