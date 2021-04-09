import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, abspath, exists
import pandas as pd


def Preprocess(img_dir):
    ''' Load the frames & locations
        Args:
            img_dir: directory of the image frames
        Return:
            image_frames: list of image frames
    '''
    image_path = abspath(img_dir)
    image_list = [join(image_path, file) for file in listdir(image_path) if isfile(join(image_path, file)) and 'tif' in file] 
    image_list.sort()

    # load data
    image_frames = []
    for file in image_list:
        img = cv2.imread(file)
        image_frames.append(img.copy())
    image_frames = np.array(image_frames)

    return image_frames
