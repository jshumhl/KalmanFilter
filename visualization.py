import cv2
import numpy as np
from os import makedirs
from os.path import exists

# Store the frames in a directory
def output_visualization(dir, frames):
    # Write the image to the output file
    if not exists(dir):
        makedirs(dir)
    
    count = 1
    for output in frames:
        cv2.imwrite(dir+str(count)+'.jpeg',output)
        count += 1

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(dir+'output_video.mp4', fourcc, 3.0, size)
    for output in frames:
        out.write(output)
    out.release()