import gradio as gr

def retrieval(text):
    return ["Hello " + text + "!", 
            "https://example.com/video1.mp4",
            "https://example.com/video2.mp4",
            "https://example.com/video3.mp4"]

with gr.Interface(fn=retrieval, 
                  inputs=gr.inputs.Textbox(label="Text", lines=2, placeholder="Text Here..."), 
                  outputs=[gr.outputs.Video(label="Output 1"), 
                           gr.outputs.Video(label="Output 2"),
                           gr.outputs.Video(label="Output 3"),
                           gr.outputs.Video(label="Output 4")]) as iface:
    iface.launch()
