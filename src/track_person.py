import numpy as np
import cv2
import os
import json
import cvxpy as cvx
from cvxopt import *
import matplotlib.pyplot as plt

FRAMES_PATH = '/ssd_scratch/cvit/pradeep/lec_frames/'
OUTPUT_DIR = '/ssd_scratch/cvit/pradeep/output/'
DARKNET_DIR = '../../darknet/'

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
cur_dir = os.getcwd()
os.chdir(DARKNET_DIR)

darknet_cmd = "./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -dont_show -ext_output < " + \
        OUTPUT_DIR+"inp_frames.txt > " + OUTPUT_DIR+"detections.txt"
os.system(darknet_cmd)

os.chdir(cur_dir)

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
	width = width + int(wnd_data[nm]['cord']['width'])
	height = height + int(wnd_data[nm]['cord']['height'])
width = width/len(frame_names)
height = height/len(frame_names)

x_original = []
y_original = []
for nm in frame_names:
	x_original.append(int(wnd_data[nm]['cord']['left_x']))
	y_original.append(int(wnd_data[nm]['cord']['top_y']))

x_optimal = find_optimal(x_original)
y_optimal = find_optimal(y_original)

'''
plt.plot(x_original)
plt.plot(x_optimal)
plt.legend(['Original', 'Final'])
plt.show()
'''

opt_cord = {}
opt_cord['avg_width'] = width
opt_cord['avg_height'] = height
for idx,nm in enumerate(frame_names):
    opt_cord[nm] = {'left_x': x_optimal[idx], 'top_y' : y_optimal[idx]}
f = open(OUTPUT_DIR+"optimal.txt","w+")
json.dump(opt_cord,f)
