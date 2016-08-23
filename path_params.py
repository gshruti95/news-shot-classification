## Paths to all required external files 

# Path to installed caffe directory
caffe_path = './mycaffe/'
pycaffe_path = caffe_path + 'python'

# SVM files
fpickle = './5class_ovr_classifier.pkl'
features_file = 'cropped_places_fc7.csv'

# Places CNN path --> mycaffe/models/placesCNN
placesCNN_path = caffe_path + 'models/placesCNN/'
placesCNN_prototxt = placesCNN_path + 'places205CNN_deploy_upgraded.prototxt'
placesCNN_caffemodel = placesCNN_path + 'places205CNN_iter_300000_upgraded.caffemodel'
placesCNN_mean = placesCNN_path + 'places205CNN_mean.npy'
places_labels = placesCNN_path + 'IndoorOutdoor_places205.csv'
scene_values = placesCNN_path + 'attributeValues.csv'
scene_names = placesCNN_path + 'attributeNames.csv'

# Googlenet path --> mycaffe/models/bvlc_googlenet
googlenet_path = caffe_path + 'models/bvlc_googlenet/'
googlenet_prototxt = googlenet_path + 'deploy.prototxt'
googlenet_caffemodel = googlenet_path + 'bvlc_googlenet.caffemodel'
imagenet_mean = googlenet_path + 'imagenet_mean.npy'
bet_pickle = googlenet_path + 'imagenet.bet.pickle'
dictionary_file = './dictionary.txt'

# Finetuned net path --> mycaffe/models/finetune
finetune_path = caffe_path + 'models/finetune/'
finetune_prototxt = finetune_path + '5class_deploy.prototxt'
finetune_caffemodel = finetune_path + '5class_newsshot_iter_100000.caffemodel'
finetune_mean = imagenet_mean
finetune_labels = finetune_path + '5class.csv'

