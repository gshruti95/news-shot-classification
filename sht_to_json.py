import sys
import json

def parse_shot_params(line,shot):
    # assignments and list of frames
    for i in range(4, len(line)):
        if '=' in line[i]:
            sub_line = line[i].split('=')
            shot[sub_line[0]] = sub_line[1]
    shot['Frames'] = []

def parse_map_of_values(line,frame):
    # map of values
    tag = line[3]
    dict = {}
    if len(line[5]) > 0:
        sub_line = line[5].replace('(', '').replace(')', '').split(', ')
        for i in range(0, len(sub_line), 2):
            dict[sub_line[i]] = float(sub_line[i + 1])
    frame[tag] = dict

def parse_finetuned_shot_class(line,frame):
    # map of values
    parse_map_of_values(line,frame)

def parse_svm_shot_class(line,frame):
    # single string
    tag = line[3]
    frame[tag] = line[4]

def parse_obj_class(line,frame):
    # map of values
    parse_map_of_values(line,frame)

def parse_scene_location(line,frame):
    # map of values
    parse_map_of_values(line,frame)

def parse_scene_attributes(line,frame):
    # list of values
    tag = line[3]
    frame[tag] = []
    sub_line = line[4].split(', ')
    for i in range(0, len(sub_line)):
        frame[tag].append(sub_line[i])

def parse_yolo_persons(line,frame):
    if len(line) <= 5:
        return
    persons = {}
    cur = line[5]
    while '[' in cur:
        cur = cur[cur.index('[')+1:]
        cur = cur[cur.index(', ')+2:]
        persons['val'] = float(cur[:cur.index(')')])
        cur = cur[cur.index('[')+1:]
        vals = cur[:cur.index(']')].split(', ')
        dict = { 'x' : int(vals[0]), 'y' : int(vals[1]),
                 'w' : int(vals[2]), 'h' : int(vals[3]) }
        persons['pos'] = dict
    frame['YOLO/PERSONS'] = persons
    pass

def sht_to_json(sht_file_name):
    frame = {}
    shot = {}
    shots = []
    time = ""
    last_time = ""
    started = False
    filename = sht_file_name
    for line in open(filename, 'r'):
        line = line.replace('\n', '').split('|')
        if len(line) > 3:
            last_time = time
            time = line[0]
        if "SHOT_DETECTED" in line:
            # shot is filled
            if 'Frames' in shot and len(shot['Frames']) > 0 or frame:
                shot['Frames'].append(frame)
                shots.append(shot)
            shot = {}
            parse_shot_params(line,shot)
            started = True
        elif started and len(line) > 3:
            # new frame
            if time != last_time:
                if frame:
                    shot['Frames'].append(frame)
                    frame = {}
            tag = line[3]
            if tag == 'FINETUNED_CLASS':
                parse_finetuned_shot_class(line,frame)
            elif tag == 'SVM_CLASS':
                parse_svm_shot_class(line,frame)
            elif tag == 'OBJ_CLASS':
                parse_obj_class(line,frame)
            elif tag == 'SCENE_LOCATION':
                parse_scene_location(line,frame)
            elif tag == 'SCENE_ATTRIBUTES':
                parse_scene_attributes(line,frame)
            elif tag == 'YOLO/PERSONS':
                parse_yolo_persons(line,frame)
            else:
                print("Unknown tag:" + tag)
                exit(1)

    with open(filename.split('.sht')[0] + '.json', 'w') as outfile:
        outfile.write("[\n")
        for i in range(0, len(shots)):
            json.dump(shots[i], outfile)
            if i != len(shots) - 1:
                outfile.write(",")
            outfile.write("\n")
        outfile.write("]")