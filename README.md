# News Shot Classification
Author: Shruti Gullapuram (gshruti95)



## Installation
Simply clone the repository, while making sure all dependencies are correctly installed

## Usage
Run `python main.py <videofilename>`

## Output
The output is stored as a file named `<videofilename>.vis` in the same directory as the video.  
It contains a variety of labels:-
- News Shot Boundaries
- News Shot Class (Newsperson, Background Roll, Graphics, Weather, Sports)
- Object Category and Labels
- Scene Location (Indoor, Outdoor) and Labels
- Scene Attributes (such as lighting)

## Dependencies
- Python (https://www.python.org/downloads/) : The language of the project
- Caffe (https://github.com/BVLC/caffe) : For neural networks
- FFMpeg (https://github.com/FFmpeg/FFmpeg) : For video processing
- PySceneDetect (https://github.com/Breakthrough/PySceneDetect) : For shot detection
- Scikit-Learn (http://scikit-learn.org/stable/install.html) : For various classifiers

## Google Summer of Code
This is the project repository for a Google Summer of Code project for Red Hen Lab.  
The project link is https://summerofcode.withgoogle.com/projects/#6049536703201280
