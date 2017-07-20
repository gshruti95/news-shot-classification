import sys
import json

filename = ""
frame = {}
shot = {}
shots = []
time = ""
last_time = ""
started = False

def parse_shot_params(line):
    # assignments and list of frames
    for i in range(4, len(line)):
        if '=' in line[i]:
            sub_line = line[i].split('=')
            shot[sub_line[0]] = sub_line[1]
    shot['Frames'] = []

def parse_map_of_values(line):
    # map of values
    tag = line[3]
    dict = {}
    if len(line[5]) > 0:
        sub_line = line[5].replace('(', '').replace(')', '').split(', ')
        for i in range(0, len(sub_line), 2):
            dict[sub_line[i]] = float(sub_line[i + 1])
    frame[tag] = dict

def parse_finetuned_shot_class(line):
    # map of values
    parse_map_of_values(line)

def parse_svm_shot_class(line):
    # single string
    tag = line[3]
    frame[tag] = line[4]

def parse_obj_class(line):
    # map of values
    parse_map_of_values(line)

def parse_scene_location(line):
    # map of values
    parse_map_of_values(line)

def parse_scene_attributes(line):
    # list of values
    tag = line[3]
    frame[tag] = []
    sub_line = line[4].split(', ')
    for i in range(0, len(sub_line)):
        frame[tag].append(sub_line[i])

def parse_yolo_persons(line):
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please write path to input file")
        exit(1)
    filename = sys.argv[1]

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
            parse_shot_params(line)
            started = True
        elif started and len(line) > 3:
            # new frame
            if time != last_time:
                if frame:
                    shot['Frames'].append(frame)
                    frame = {}
            tag = line[3]
            if tag == 'FINETUNED_SHOT_CLASS':
                parse_finetuned_shot_class(line)
            elif tag == 'SVM_SHOT_CLASS':
                parse_svm_shot_class(line)
            elif tag == 'OBJ_CLASS':
                parse_obj_class(line)
            elif tag == 'SCENE_LOCATION':
                parse_scene_location(line)
            elif tag == 'SCENE_ATTRIBUTES':
                parse_scene_attributes(line)
            elif tag == 'YOLO/PERSONS':
                parse_yolo_persons(line)
            else:
                print("Unknown tag:" + tag)
                exit(1)

    with open(filename[:filename.index('.')] + '.json', 'w') as outfile:
        outfile.write("[\n")
        for i in range(0, len(shots)):
            json.dump(shots[i], outfile)
            if i != len(shots) - 1:
                outfile.write(",")
            outfile.write("\n")
        outfile.write("]")