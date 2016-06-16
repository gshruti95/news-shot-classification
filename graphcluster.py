import os, sys, time	
import numpy as np
import scipy
from skimage import io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import KMeans
import facedetect
import age_genderCNN, fileops

def get_hist(image):

	img = io.imread(image)
	h = img.shape[0]
	w = img.shape[1]
	
	sub_ims = []
	hist = []
	hists_r = []
	hists_g = []
	hists_b = []
	for i in xrange(4):
		for j in xrange(4):
			sub_im = img[j*h/4:(j*h/4)+h/4,i*w/4:(i*w/4)+w/4]
			sub_ims.append ( sub_im )
			hist_r = np.histogram(sub_im[:,:,0])
			hist_g = np.histogram(sub_im[:,:,1])
			hist_b = np.histogram(sub_im[:,:,2])
			hists_r.append(hist_r[0])
			hists_g.append(hist_g[0])
			hists_b.append(hist_b[0])
			
	hist.append(hists_r)
	hist.append(hists_g)
	hist.append(hists_b)

	return hist

def get_edge(hist1, hist2):

	hod_r = abs(np.asarray(hist1[0]) - np.asarray(hist2[0]))
	hod_g = abs(np.asarray(hist1[1]) - np.asarray(hist2[1]))
	hod_b = abs(np.asarray(hist1[2]) - np.asarray(hist2[2]))

	hod = hod_r + hod_g + hod_b	
	hod = np.sum(hod, axis=1)

	edge_weight = sum(np.sort(hod)[0:8])
	return edge_weight

def get_mst(adj_mat):
	X = csr_matrix(adj_mat)
	# print adj_mat
	Tcsr = minimum_spanning_tree(X)
	mst = Tcsr.toarray().astype(int)
	weights = []
	for row in mst:
		for value in row[np.nonzero(row)]:
			weights.append(value)
	# print weights
	return mst, weights

def get_cluster_threshold(weights):
	
	estimator = KMeans(n_clusters = 2)
	data = np.asarray(weights)
	data = data.reshape(-1,1)
	# print data 
	clusters_idx = estimator.fit_predict(data)
	max_idx = data.argmax()
	max_cluster = clusters_idx[max_idx]
	#print max_cluster
	low_cluster = []
	if max_cluster == 1:
		indices = np.argwhere(clusters_idx == 0)
		for idx in indices:
			low_cluster.append(data[idx])
		threshold = max(low_cluster)
		threshold = threshold[0][0]
	else:
		indices = np.argwhere(clusters_idx == 1)
		for idx in indices:
			low_cluster.append(data[idx])
		threshold = max(low_cluster)
		threshold = threshold[0][0]
	# print threshold
	return threshold

def get_graph_clusters(clip_dir, image_files):

	start = time.time()

	hists = []
	adj_mat = []
	print "Calculating histograms..."
	for image in image_files:
		new_hist = get_hist(image)
		hists.append(new_hist)
	
	print "Calculating edge weights..."
	for i in xrange(len(hists)):
		row = []
		for j in xrange(len(hists)):
			if i < j:
				edge = get_edge(hists[i],hists[j])
			else:
				edge = 0
			row.append(edge)
		adj_mat.append(row)

	mst, weights = get_mst(adj_mat)
	print "MST created..."
	threshold = get_cluster_threshold(weights)
	mst[np.where(mst > threshold)] = 0	
	print "Clusters formed..."
	# print len(connected_components(mst)[1]), len(image_files)
	# for i in xrange(len(image_files)):
		# print connected_components(mst, directed = False)[1][i], image_files[i].split('/')[-1]

	n_components, components = connected_components(mst, directed = False)	
	for i in xrange(n_components):
		component_indices = np.argwhere(components == i)
		# print type(component_indices)
		# indices = np.asarray(component_indices.tolist())
		cluster_faces = []
		cluster_lifetime = max(component_indices) - min(component_indices) + 1

		print "Pruning clusters..."
		if cluster_lifetime >= 50 and len(component_indices) >= 5:
			
			for i in xrange(len(component_indices)):
				cluster_faces.append(image_files[component_indices[i][0]])

			faces_count, single_faces, single_faces_frameno = facedetect.get_faces(clip_dir, cluster_faces, component_indices)		
			print "Getting faces..."
			print np.mean(faces_count)

			if np.mean(faces_count) >= 1: # At least one face detected in the cluster per image
				print 'Candidate cluster: '#, cluster_lifetime, len(component_indices)
				print component_indices
				# print cluster_faces
				print "Single faces len, face count len, img files len: \n", len(single_faces_frameno), len(faces_count), len(image_files)
				# print "Singles faces no and names: ", single_faces_frameno, single_faces 
				## Get gender labels for single face frames
				print "Getting gender labels..."
				caffe_path = '/home/shruti/gsoc/caffehome/caffe/' 
				[age_labels, gender_labels] = age_genderCNN.age_genderCNN(caffe_path, caffe_path + 'models/age_gender/', single_faces)
				print "Saving gender labels..."
				output_filename = clip_dir.split('/')[-2]	
				fileops.save_age_gender_labels(clip_dir + output_filename, clip_dir + 'age_gender_labels', age_labels, gender_labels, single_faces_frameno, faces_count, component_indices)
			else:
				print 'Cluster Rejected: ', cluster_lifetime, len(component_indices)
				print cluster_faces[0]

	end = time.time()
	print "Time taken for clusters and genders: %.2f \n" %(end-start)


 # get_hist('/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0002.jpg')
# get_graph_clusters(['/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0002.jpg',
#  '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0003.jpg',
#  '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0004.jpg',
#  '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0284.jpg'])