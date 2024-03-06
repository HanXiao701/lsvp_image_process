import cv2
import json
import segeval as se
import numpy as np


def convert_to_mass_format(segment_points):

    boundaries = []
    for i,f in enumerate(segment_points):
        if i==0:
            boundaries.append(segment_points[i])

        else:
            boundaries.append(segment_points[i]-segment_points[i-1])

    return boundaries

def convert_to_segments(video_name, video_type):

    video_path = "./assets/videos/{}.{}".format(video_name, video_type)
    print (video_path)
    video = cv2.VideoCapture(video_path)

    # Check if file opened successfully
    assert video.isOpened()

    # video meta-data
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_video_time = int(frame_count/fps)

    manual_segments = []
    with open("./assets/boundaries/{}.txt".format(video_name)) as file:
        for line in file:
            manual_segments.append(int(line.rstrip()))

    manual_segments.append(max_video_time)



    manual_boundaries = convert_to_mass_format(manual_segments)
    avg_manual_segment_length = int(np.mean(manual_boundaries))


    detection = json.load(open("./assets/json/{}_objects.json".format(video_name)))
    automatic_segments = []
    for i in detection:
        if int(i) == len(detection)-1:
            break

        automatic_segments.append(int(int(detection[i]['start'])/fps))
        automatic_segments.append(int(int(detection[i]['end'])/fps))
    print (automatic_segments)

    automatic_segments = sorted([*set(automatic_segments)])
    automatic_segments.append(max_video_time)

    automatic_boundaries = convert_to_mass_format(automatic_segments)

    uniform_segments = [ i for i in range(1, max_video_time) if i%4 == 0 ]
    uniform_segments.append(max_video_time)

    uniform_boundaries = convert_to_mass_format(uniform_segments)

    
    print (manual_segments)
    print (manual_boundaries)
    print (avg_manual_segment_length)


    bs_automatic = se.boundary_string_from_masses(tuple(automatic_boundaries))
    bs_manual = se.boundary_string_from_masses(tuple(manual_boundaries))
    bs_uniform = se.boundary_string_from_masses(tuple(uniform_boundaries))

    b = se.boundary_similarity(bs_automatic, 
                                bs_manual, 
                                boundary_format=se.BoundaryFormat.sets, 
                                n_t=4)

    b_uniform = se.boundary_similarity(bs_uniform, 
                                bs_manual, 
                                boundary_format=se.BoundaryFormat.sets, 
                                n_t=4)

    print (b)
    print (b_uniform)



if __name__ == '__main__':
    video_name = "bio"  # input video name here
    video_type = "mp4"   # generally mp4

    # Load video (only needed once, comment out once loaded)
    print ("video name: {}.{}".format(video_name, video_type))
    convert_to_segments(video_name, video_type)
