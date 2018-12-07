import numpy as np
import cv2
import os
import json
import cvxpy as cvx
from cvxopt import *
import matplotlib.pyplot as plt

FRAMES_PATH = '../data/01/video_frames/'
OUTPUT_DIR = '../data/01/output/'
DARKNET_DIR = './'

def find_optimal(signal, W = [10,1,200]):
	out=cvx.Variable(len(signal))
	
	c1=0.0
	for i in range(len(signal)):
		c1+=(signal[i]-out[i])**2

	c2=0.0
	diff=[]
	for i in range(1,len(signal)):
		c2+=(out[i]-out[i-1])**2
		diff.append(out[i]-out[i-1])

	c3=0.0
	for i in range(1,len(diff)):
		c3+=(diff[i]-diff[i-1])**2
	
	cost= W[0]*c1 + W[1]*c2 + W[2]*c3
	problem = cvx.Problem(cvx.Minimize(cost))
	problem.solve()
	return out.value

# Reading the frame names in sorted order
frame_names = [os.path.abspath(os.path.join(FRAMES_PATH, p)) for p in os.listdir(FRAMES_PATH)]
frame_names.sort(key=lambda f: int(filter(str.isdigit, f.split('/')[-1].lower())))

# Writing the frame names into inp_frames.txt
f = open(OUTPUT_DIR+"inp_frames.txt","w+")
for nm in frame_names:
	f.write(nm+"\n")
f.close()

# Run YOLO and save the detections into detections.txt file
darknet_cmd = DARKNET_DIR + "darknet detector test " + DARKNET_DIR + "cfg/coco.data " + DARKNET_DIR + "cfg/yolov3.cfg " + \
				DARKNET_DIR + "yolov3.weights -dont_show -ext_output < " + OUTPUT_DIR+"inp_frames.txt > " + OUTPUT_DIR+"detections.txt"
# os.system(darknet_cmd)

# Load the data of the window co-ordinates
f = open(OUTPUT_DIR+"detections.txt","r")
f.readline()
f.readline()
f.readline()
wnd_data = json.load(f)

# Get the average as the height and width of the window
width = 0
height = 0
for nm in frame_names:
	width = width + wnd_data[nm]['cord']['width']
	height = height + wnd_data[nm]['cord']['height']
width = width/len(frame_names)
height = height/len(frame_names)

x_orignal = []
y_orignal = []
for nm in frame_names:
	x_orignal.append(int(wnd_data[nm]['cord']['left_x']))
	y_orignal.append(int(wnd_data[nm]['cord']['top_y']))

x_optimal = find_optimal(x_orignal)
y_optimal = find_optimal(y_orignal)