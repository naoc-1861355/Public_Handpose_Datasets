import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random
from shutil import copy, copyfile
import json


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


def viz(inpath, outpath):
    """
    normdat is a function that convert this dataset to standard ezxr format output

    Args:
        :param inpath: path to this dataset
        :param outpath: output path of the formatted files
    Returns:
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
            max_x = 0
            min_x = 99999
            max_y = 0
            min_y = 99999
            for k in range(len(pts2d)):
                p = pts2d[k][0]
                if p[0] > max_x:
                    max_x = p[0]
                if p[0] < min_x:
                    min_x = p[0]
                if p[1] > max_y:
                    max_y = p[1]
                if p[1] < min_y:
                    min_y = p[1]
            hand_bbox = [min_x, max_x, min_y, max_y]

            f52 = {'x': -1, 'y': -1, 'd': -1}
            f12 = [{'x': pts2d[1, 0, 0], 'y': pts2d[1, 0, 1], 'd': -1},
                   {'x': pts2d[0, 0, 0], 'y': pts2d[0, 0, 1], 'd': -1},
                   {'x': pts2d[2, 0, 0], 'y': pts2d[2, 0, 1], 'd': -1},
                   {'x': pts2d[3, 0, 0], 'y': pts2d[3, 0, 1], 'd': -1}]
            f22 = [{'x': pts2d[5, 0, 0], 'y': pts2d[5, 0, 1], 'd': -1},
                   {'x': pts2d[4, 0, 0], 'y': pts2d[4, 0, 1], 'd': -1},
                   {'x': pts2d[6, 0, 0], 'y': pts2d[6, 0, 1], 'd': -1},
                   {'x': pts2d[7, 0, 0], 'y': pts2d[7, 0, 1], 'd': -1}]
            f42 = [{'x': pts2d[9, 0, 0], 'y': pts2d[9, 0, 1], 'd': -1},
                   {'x': pts2d[8, 0, 0], 'y': pts2d[8, 0, 1], 'd': -1},
                   {'x': pts2d[10, 0, 0], 'y': pts2d[10, 0, 1], 'd': -1},
                   {'x': pts2d[11, 0, 0], 'y': pts2d[11, 0, 1], 'd': -1}]
            f32 = [{'x': pts2d[13, 0, 0], 'y': pts2d[13, 0, 1], 'd': -1},
                   {'x': pts2d[12, 0, 0], 'y': pts2d[12, 0, 1], 'd': -1},
                   {'x': pts2d[14, 0, 0], 'y': pts2d[14, 0, 1], 'd': -1},
                   {'x': pts2d[15, 0, 0], 'y': pts2d[15, 0, 1], 'd': -1}]
            f02 = [{'x': pts2d[17, 0, 0], 'y': pts2d[17, 0, 1], 'd': -1},
                   {'x': pts2d[16, 0, 0], 'y': pts2d[16, 0, 1], 'd': -1},
                   {'x': pts2d[18, 0, 0], 'y': pts2d[18, 0, 1], 'd': -1},
                   {'x': pts2d[19, 0, 0], 'y': pts2d[19, 0, 1], 'd': -1}]

            f53 = {'x': -1, 'y': -1, 'd': -1}
            f13 = [{'x': points3d[1, 0], 'y': points3d[1, 1], 'z': points3d[1, 2]},
                   {'x': points3d[0, 0], 'y': points3d[0, 1], 'z': points3d[0, 2]},
                   {'x': points3d[2, 0], 'y': points3d[2, 1], 'z': points3d[2, 2]},
                   {'x': points3d[3, 0], 'y': points3d[3, 1], 'z': points3d[3, 2]}]
            f23 = [{'x': points3d[5, 0], 'y': points3d[5, 1], 'z': points3d[5, 2]},
                   {'x': points3d[4, 0], 'y': points3d[4, 1], 'z': points3d[4, 2]},
                   {'x': points3d[6, 0], 'y': points3d[6, 1], 'z': points3d[6, 2]},
                   {'x': points3d[7, 0], 'y': points3d[7, 1], 'z': points3d[7, 2]}]
            f43 = [{'x': points3d[9, 0], 'y': points3d[9, 1], 'z': points3d[9, 2]},
                   {'x': points3d[8, 0], 'y': points3d[8, 1], 'z': points3d[8, 2]},
                   {'x': points3d[10, 0], 'y': points3d[10, 1], 'z': points3d[10, 2]},
                   {'x': points3d[11, 0], 'y': points3d[11, 1], 'z': points3d[11, 2]}]
            f33 = [{'x': points3d[13, 0], 'y': points3d[13, 1], 'z': points3d[13, 2]},
                   {'x': points3d[12, 0], 'y': points3d[12, 1], 'z': points3d[12, 2]},
                   {'x': points3d[14, 0], 'y': points3d[14, 1], 'z': points3d[14, 2]},
                   {'x': points3d[15, 0], 'y': points3d[15, 1], 'z': points3d[15, 2]}]
            f03 = [{'x': points3d[17, 0], 'y': points3d[17, 1], 'z': points3d[17, 2]},
                   {'x': points3d[16, 0], 'y': points3d[16, 1], 'z': points3d[16, 2]},
                   {'x': points3d[18, 0], 'y': points3d[18, 1], 'z': points3d[18, 2]},
                   {'x': points3d[19, 0], 'y': points3d[19, 1], 'z': points3d[19, 2]}]

            # show a random sample of the sequence
            dict_kp = {'palm_center': points3d[20, :].tolist(), 'is_left': -1, 'hand_bbox': hand_bbox,
                       'f52': [f52 for _ in range(4)], 'f53': [f53 for _ in range(4)],
                       'f03': f03, 'f13': f13, 'f23': f23, 'f33': f33, 'f43': f43,
                       'f02': f02, 'f12': f12, 'f22': f22, 'f32': f32, 'f42': f42}
            outpath = outputdir + colorFrames[j][19:-4]
            outpath_img = outpath + '.jpg'
            outpath_json = outpath + '.json'
            copyfile(colorFrames[j], outpath_img)
            with open(outpath_json, 'w') as outfile:
                json.dump(dict_kp, outfile)


def main():
    pathToDataset = "../annotated_frames/"
    outpath = "../out_2"
    viz(pathToDataset, outpath)


if __name__ == "__main__":
    main()
