# News Shot Classification
Author: Shruti Gullapuram (gshruti95)

## Installation
Clone the repository, while ensuring that all dependencies are correctly installed.

## Usage
Run `python ShotClass-01.py <path-to-videofile>`	
The path to the video file is either absolute or relative with respect to the `ShotClass-01.py` file.

### Usage on Case HPC
- Process a list of videos using -l flag:
	Run `./manager.sh -l <list>.txt`
	<list>.txt contains YYYY-MM-DD_HOUR_NETWORKNAME.mp4 (only basenames of files)
- Process a particular day's worth of news videos using -d flag:
	Run `./manager.sh -d YYYY/MM/DD`
	Run `./manager.sh <path-to-videofile>`
- You can edit the variable VIDEO_DST in manager.sh to change the path of the processed video files.

## Output
The output is stored as two files named `<videofilename>.sht` and `<videofilename>.json` (json lines format) in the same directory as the video.   
- Camera shot type → [ Newsperson(s), Background_roll, Graphic, Weather, Sports ]
- Object category → [ Vehicle, Natural formation, Building/Structure, Person(s)/Clothing, Weapon, Sports ]
- Scene type → [ Indoor, Outdoor ]
- Imagenet labels with probabilities
- Places205 labels with probabilities
- Scene attributes
- YOLO/Persons with detected count, probability and position of each detection as x,y coordinates and height and width.

## Dependencies
- Python (https://www.python.org/downloads/) : The language of the project. The code has been tested with Python 2.7.8 and 2.7.12. It should work with any recent version of Python 2.
- Caffe (https://github.com/BVLC/caffe) : Neural network framework. Needs to be complied with Python support (pycaffe). The code has been tested with Caffe rc3, and should work with the GitHub version.
- FFMpeg (https://github.com/FFmpeg/FFmpeg) : For video processing. The code has been tested with v2.8.2 and v3.1.0, and should work with the GitHub version.
- PySceneDetect (https://github.com/Breakthrough/PySceneDetect) : For shot detection. The code has been tested with v0.3.5.
- Scikit-Learn (http://scikit-learn.org/stable/install.html) : For various classifiers

### Required External Files and Models
All the required external files and classifier models can be found here:
https://www.dropbox.com/sh/hv811iqnupcusp8/AAA-nn4mYD2LIP2-deK1VUSWa?dl=0	
The paths to all external files required by the code can be modified in `path_params.py` according to the user’s convenience.

## Google Summer of Code
This is the project repository for a Google Summer of Code 2016 project for Red Hen Lab.  
The project link is https://summerofcode.withgoogle.com/projects/#6049536703201280	 
The final work product submission is at https://shrutigullapuram.wordpress.com/2016/08/22/gsoc-work-product-submission/

## Citations/Licenses
- Places205-AlexNet model: 
	B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva
	Learning Deep Features for Scene Recognition using Places Database.
	Advances in Neural Information Processing Systems 27 (NIPS) spotlight, 2014.
	http://places.csail.mit.edu/downloadCNN.html
- GoogleNet model:
	http://arxiv.org/abs/1409.4842
	Szegedy et al., Going Deeper with Convolutions, CoRR 2014
	Used BVLC Googlenet model, trained by S. Guadarama.
- Reference Caffenet model:
	AlexNet trained on ILSVRC 2012, with a minor variation from the version as described in ImageNet classification with deep convolutional neural networks by Krizhevsky et al. in NIPS 2012. Model trained by J. Donahue.
- Red Hen Lab NewsScape Dataset:
	This work made use of the NewsScape dataset and the facilities of the Distributed Little Red Hen Lab, co-directed by Francis Steen and Mark Turner.
	http://redhenlab.org
- YOLO model:
	https://arxiv.org/abs/1506.02640
	Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, You Only Look Once: Unified, Real-Time Object Detection, CoRR 2015
	A demo of the actual system and the source code can be found on their project website: http://pjreddie.com/yolo/
