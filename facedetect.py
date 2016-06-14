import os, sys
import dlib
from skimage import io

def get_faces(clip_dir, image_files):

	detector = dlib.get_frontal_face_detector()
	win = dlib.image_window()

	faces_count = []
	faces = []
	faces_frameno = []
	frame = 0
	temp_dir = clip_dir + 'temp/'
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)

	for image in image_files:
		frame += 1
		print("Processing file: {}".format(image))
		img = io.imread(image)
		# The 1 in the second argument indicates that we should upsample the image
		# 1 time.  This will make everything bigger and allow us to detect more
		# faces.
		dets = detector(img, 1)
		print("Number of faces detected: {}".format(len(dets)))
		faces_count.append(len(dets))

		if 0 < len(dets) < 2:
			faces_frameno.append(frame)
			img_cropped = img[dets[0].top():dets[0].bottom(), dets[0].left():dets[0].right()]
			io.imsave(temp_dir + str(frame) + '.jpg', img_cropped)
			faces.append(temp_dir + str(frame) + '.jpg')

		# for i, d in enumerate(dets):
		# 	print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
		# 		i, d.left(), d.top(), d.right(), d.bottom()))

		win.clear_overlay()
		win.set_image(img)
		win.add_overlay(dets)
		
		#if (len(sys.argv[1:]) > 0):
		#	img = io.imread(sys.argv[1])
		dets, scores, idx = detector.run(img, 1)
		for i, d in enumerate(dets):
			print("Detection {}, score: {}, face_type:{}".format(
				d, scores[i], idx[i]))
		
		#dlib.hit_enter_to_continue()
	return faces_count, faces, faces_frameno
		