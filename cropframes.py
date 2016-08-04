import os, sys
import time
import numpy as np
from skimage import io


def cropframes(clip_dir, image_files):

	crop_dir = clip_dir + 'cropped/'
	if not os.path.exists(crop_dir):
		os.makedirs(crop_dir)

	cropped_files = []

	for idx, image in enumerate(image_files):
		
		img = io.imread(image)
		h = img.shape[0]
		w = img.shape[1]
		img_cropped = img[0:4*h/5, 0:w]
		io.imsave(crop_dir + 'keyframe' + str(idx) + '.jpg', img_cropped)
		cropped_files.append(crop_dir + 'keyframe' + str(idx) + '.jpg')

	return cropped_files