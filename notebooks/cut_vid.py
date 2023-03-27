import os
import ffmpeg
import csv
import pandas as pd

#this code (using .scene.txt files found by make_scene_files.py) trims all the scenes 

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



df_array = []

#the path to the directory of .scene.txt files
for path in os.listdir("/srv/home/ahmadi/gitrepos/video-ir/scene_txt"):
    print(path)
    path2 = os.path.join("/srv/home/ahmadi/gitrepos/video-ir/scene_txt", path)
    # print(path2)

    all_scenes = []

    #finding all start and end scenes
    with open(path2, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            temp = row[0].split(" ")
            temp[0] = int(temp[0])
            temp[1] = int(temp[1])
            # print(temp)
            all_scenes.append(temp)

        #cutting each scene from the video in path2
        path = path[:13]
        for i in all_scenes:
            
            #the path to the directory of videos (that are used to make .scene.txt files previously)
            path3 = os.path.join("/raid/datasets/msr-vtt/Test2", path)
            save_path = "/srv/home/ahmadi/gitrepos/video-ir/scene_mp4/"+ str(i[0]) + "-" + str(i[1]) + "-" +  path

            temp2 = []
            temp2.append(save_path)
            temp2.append(path)
            temp2.append(i[0])
            temp2.append(i[1])
            # print(temp2)

            df_array.append(temp2)

            print(temp2)

            trim(path3, save_path , i[0], i[1])

    print("*******************************************************************************************************************")
    print("*******************************************************************************************************************")

    
df = pd.DataFrame(df_array, columns=['scene path', 'original video', 'start frame', 'end_frame'])
df.to_csv('/srv/home/ahmadi/gitrepos/video-ir/data-1.csv')