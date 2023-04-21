import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import ffmpeg

from transnetv2 import TransNetV2

def trim(in_file, out_file, start, end):
    print(in_file)
    if os.path.exists(out_file):
        os.remove(out_file)

    in_file_probe_result = ffmpeg.probe(in_file)

    input_stream = ffmpeg.input(in_file, noaccurate_seek=None)

    pts = "PTS-STARTPTS"
    video = input_stream.trim(start_frame=start, end_frame=end).setpts(pts)

    audio_stream = next((s for s in in_file_probe_result["streams"] if s["codec_type"] == "audio"), None)
    if audio_stream:
        audio = (input_stream
                 .filter_("atrim", start=start, end=end)
                 .filter_("asetpts", pts))
        video_and_audio = ffmpeg.concat(video, audio, v=1, a=1)
    else:
        video_and_audio = video

    output = ffmpeg.output(video_and_audio, out_file, format="mp4")
    output.run()


def main(arr, p_save, error_path):
    import sys
    # import argparse
    flag = False

    # model = TransNetV2(args.weights)
    model = TransNetV2(None)

    er = open(error_path, 'a+')
    er.seek(0)

    f= []
    for x in os.listdir(p_save):
        p =  os.path.join(p_save, x)
        f.append(p)

    # Iterate over each video file
    for file in tqdm(arr, desc="Iterate over each video file..."):

        flag = False
        if file in  f:
            flag = True


        if flag == False:

            video_frames, single_frame_predictions, all_frame_predictions = \
                model.predict_video(file)


            scenes = model.predictions_to_scenes(single_frame_predictions)


            #cutting each scene from the video in path2
            for i in tqdm(scenes, desc=f"cutting each scene from the video in {file}"):
                #the path to the directory of videos (that are used to make .scene.txt files previously)
                path3 = file
                ind = file.rfind("/")
                path = file[ind+1:]

                #the path to save scenes.mp4
                save_path = p_save+ str(i[0]) + "-" + str(i[1]) + "-" +  path

                temp2 = []
                temp2.append(save_path)
                temp2.append(path)
                temp2.append(i[0])
                temp2.append(i[1])

                #check if the number of frames is more than 10
                if (i[1]-i[0]) > 9:

                    try:
                        trim(path3, save_path , i[0], i[1])
                    except:
                        er.seek(0)
                        er.write(file+"\n")

#path to video files
video_repos = "/raid/datasets/msr-vtt/TrainValVideo"

#path to save scene files
p_save = "/raid/datasets/msr-vtt/Train_scene_mp4/"

#path to error paths
error_path = "/srv/home/ahmadi/gitrepos/video-ir/error_paths.txt"


path_array = []
for path in os.listdir(video_repos):
    path = os.path.join(video_repos, path)
    path_array.append(path)

main(path_array, p_save, error_path)