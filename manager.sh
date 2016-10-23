#!/bin/bash

SSH_LOC=turnerstudents@cartago
LOGS_DIR=/home/sxg755/logs
VIDEO_SRC=/tv
VIDEO_DST=/home/sxg755/videos

case "$1" in
	-d) 		#./manager.sh -d YYYY/MM/DD

	DATE="$2"
	IFS='/' read -a datearray <<< "$DATE"

	YEAR="${datearray[0]}"
	MONTH="${datearray[1]}"
	DAY="${datearray[2]}"

	SRC_DAY_LIST=$(rsync $SSH_LOC "ls -1 $VIDEO_SRC/$YEAR/$YEAR-$MONTH/$YEAR-$MONTH-$DAY/*.mp4")

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
		rsync $SSH_LOC:$f $VIDEO_DST
		sbatch process_video.slurm $VIDEO_DST/$(basename $f)
	done
	;;

	-l) 		# ./manager.sh -l files.txt
				# files.txt contains YYYY-MM-DD_HOUR_NETWORKNAME.mp4 (only basenames of files)
	
	VIDEO_DST=$VIDEO_DST/"processed_list"

	if ! [ -d "$VIDEO_DST" ]; then
		rm -rf $VIDEO_DST
	fi
	mkdir -p $VIDEO_DST
	
	while read -r VIDEONAME || [[ -n "$VIDEONAME" ]]; do
    IFS='_' read -a videodate <<< "$VIDEONAME"
    IFS='-' read -a datearray <<< "${videodate[0]}"

    YEAR="${datearray[0]}"
	MONTH="${datearray[1]}"
	DAY="${datearray[2]}"

	echo "Processing $VIDEONAME"

	rsync $SSH_LOC:$VIDEO_SRC/$YEAR/$YEAR-$MONTH/$YEAR-$MONTH-$DAY/$VIDEONAME $VIDEO_DST
	sbatch process_video.slurm $VIDEO_DST/$VIDEONAME

	done < "$2"

	;;

esac