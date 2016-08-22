#!/bin/bash

SSH_LOC=turnerstudents@cartago
VIDEO_SRC=/tv
VIDEO_DST=/home/sxg755/videos
LOGS_DIR=/home/sxg755/logs

YEAR=2016
MONTH=08
DAY=03

SRC_DAY_LIST=$(ssh $SSH_LOC "ls -1 $VIDEO_SRC/$YEAR/$YEAR-$MONTH/$YEAR-$MONTH-$DAY/*.mp4")

#if [ ! -d "$LOGS_DIR/$YEAR/$YEAR-$MONTH/$YEAR-$MONTH-$DAY" ]; then
#	mkdir -p $LOGS_DIR/$YEAR/$YEAR-$MONTH/$YEAR-$MONTH-$DAY
#fi

echo "Processing $(echo "$SRC_DAY_LIST" | wc -l) files for $YEAR-$MONTH-$DAY"

VIDEO_DST=$VIDEO_DST/$YEAR/$YEAR-$MONTH/$YEAR-$MONTH-$DAY

if [ -d "$VIDEO_DST" ]; then
	rm -rf $VIDEO_DST
fi
mkdir -p $VIDEO_DST

for f in $SRC_DAY_LIST;do 
	FILENAME=$(basename $f)
	scp $SSH_LOC:$f $VIDEO_DST
	sbatch process_video.slurm $VIDEO_DST/$(basename $f)
done

# Tracking the jobs

# Handling the failed jobs


