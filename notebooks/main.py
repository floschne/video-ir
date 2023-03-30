import find_similarity as fs
import numpy as np
import text_embedding 

import gradio as gr

def retrieval(text):

    txt = np.load('/srv/home/ahmadi/gitrepos/video-ir/car_s.npy') 
    # txt = text_embedding.find_embedding(text)

    arr = fs.find_sim(txt)

    ret_arr = []
    path = "/srv/home/ahmadi/gitrepos/video-ir/scene_mp4/"

    for i in range(len(arr)):
        x = arr[i+1][0]
        s = x.find("-")
        e = x.find(".")
        vid_name = x[s+1:e]
        ret_arr.append(vid_name)
        path_i = path + vid_name + ".mp4"
        ret_arr.append(path_i)

    return ret_arr


with gr.Interface(fn=retrieval,
                  inputs=gr.inputs.Textbox(label="Text", lines=2, placeholder="Text Here..."),
                  outputs=[gr.outputs.Textbox(label="Output 1"),
                           gr.outputs.Video(label="Output 1"),
                           gr.outputs.Textbox(label="Output 2"),
                           gr.outputs.Video(label="Output 2"),
                           gr.outputs.Textbox(label="Output 3"),
                           gr.outputs.Video(label="Output 3"),
                           gr.outputs.Textbox(label="Output 4"),
                           gr.outputs.Video(label="Output 4")]) as iface:
    iface.launch()


