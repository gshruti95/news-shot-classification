import os, sys	
import numpy as np
import scipy
from skimage import io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import KMeans
import facedetect

def get_hist(image):

	img = io.imread(image)
	h = img.shape[0]
	w = img.shape[1]
	# print (w*h/16)
	# print img[0][0] #[r g b]
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
			# print sum(hist_r[0])
			# print sum(hist[0])
	hist.append(hists_r)
	hist.append(hists_g)
	hist.append(hists_b)
	# print hist[0],len(hist[0])
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

	hists = []
	adj_mat = []
	for image in image_files:
		new_hist = get_hist(image)
		hists.append(new_hist)
	
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
	threshold = get_cluster_threshold(weights)
	mst[np.where(mst > threshold)] = 0	

	# print len(connected_components(mst)[1]), len(image_files)

	# for i in xrange(len(image_files)):
		# print connected_components(mst, directed = False)[1][i], image_files[i].split('/')[-1]

	n_components, components = connected_components(mst, directed = False)
	for i in xrange(n_components):
		indices = np.argwhere(components == i)
		# print type(indices)
		# indices = np.asarray(indices.tolist())
		faces_list = []
		cluster_lifetime = max(indices) - min(indices) + 1
		if cluster_lifetime >= 50 and len(indices) >= 5:
			for i in xrange(len(indices)):
				faces_list.append(image_files[indices[i][0]])

			faces_count, faces, faces_frameno = facedetect.get_faces(clip_dir, faces_list)
			print np.mean(faces_count)
			if np.mean(faces_count) >= 1: # At least one face detected in the cluster per image
				print 'Candidate cluster: ',cluster_lifetime, len(indices)
				print faces_list
			else:
				print 'Cluster Rejected: ', cluster_lifetime, len(indices)
				print faces_list[0]



 # get_hist('/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0002.jpg')
# get_graph_clusters(['/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0002.jpg',
#  '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0003.jpg',
#  '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0004.jpg',
#  '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/keyframe0284.jpg'])