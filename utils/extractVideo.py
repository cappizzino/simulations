import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        # save frame as JPEG file
        cv2.imwrite( pathOut + "frame%d.jpg" % count, image)     
        count = count + 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()

    args.pathIn = "/media/cappizzino/OS/Users/cappi/Documents/Estudos/04_Doutorado/Tese/datasets/videos/irat_red/log_irat_red.avi"
    args.pathOut = "./data/ratSlam1/"

    extractImages(args.pathIn, args.pathOut)