## job_manager.py - Core manager script for visual recognition processing over the entire NewsScape dataset

import sys
import os
import re
import time
import shutil

# Usage - Keep it running on Case HPC / Erlangen HPC 

# Case HPC Paths
SSH_LOC=turnerstudents@cartago
LOGS_DIR=/home/sxg755/logs
VIDEO_SRC=/tv
VIDEO_DST=/home/sxg755/videos

# Keep track of completed jobs in a stable location

# Keep track of currently available resources (CPU/GPU nodes)

# Create a new instance per single file to be processed

# Split up the various components of processing the file into separate jobs based on available resources