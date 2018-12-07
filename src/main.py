import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import os


FRAMES_PATH = '/ssd_scratch/cvit/pradeep/lec_frames/'
OUTPUT_DIR = '/ssd_scratch/cvit/pradeep/output/'

# Reading the frame names in sorted order
frame_names = [os.path.abspath(os.path.join(FRAMES_PATH, p)) for p in os.listdir(FRAMES_PATH)]
frame_names.sort(key=lambda f: int(filter(str.isdigit, f.split('/')[-1].lower())))

f = open(OUTPUT_DIR+"optimal.txt")
person_cord = json.load(f)

width = person_cord['avg_width']
height = person_cord['avg_height']

for nm in frame_names:
    frm = cv2.imread(nm)
    x = person_cord[nm]['left_x']
    y = person_cord[nm]['top_y']

    out = frm[int(y):int(y+0.75*height),int(x):int(x+width)]

    name = nm.split('/')[-1].lower()
    print(OUTPUT_DIR+"person/"+name)
    cv2.imwrite(OUTPUT_DIR+"person/"+name,out)
