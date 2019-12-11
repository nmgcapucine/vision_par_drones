import camera
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


# load some images
dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(dirname, 'Datasets')

# Image name between 3434 and 3444 for tea_scene
filename = os.path.join(data_path, 'merton1/001.JPG')
im1 = np.array(np.Image.open(filename))
filename = os.path.join(data_path, 'merton1/002.JPG')
im2 = np.array(np.Image.open(filename))
filename = os.path.join(data_path, 'merton1/003.JPG')
im3 = np.array(np.Image.open(filename))

# load 2D points for each view to a list
points2D = [np.loadtxt('2D/00')]

# load 3D points
points3D = np.loadtxt('3D/p3d').T

# load correspondences
corr = np.genfromtxt('2D/nview-corners', dtype='int',missing='*')

# load cameras to a list of Camera objects
P = [camera.Camera(np.loadtxt('2D/00'+str(i+1)+'.P')) for i in range(3)]