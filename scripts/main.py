from transformers import AutoTokenizer, AutoModel
import faiss
import glob
import gradio as gr
import fire


def find_text_embedding(text_str, mod_path):
    tokenizer = AutoTokenizer.from_pretrained(mod_path)
    model = AutoModel.from_pretrained(mod_path)

    inputs = tokenizer([text_str], padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)

    np_arr = text_features.cpu().detach().numpy()
    return np_arr


def find_sim(txt_array, ind_pa, embedding_pa):
    query_vector = txt_array
    k = 5
    index_path = ind_pa + "/Test_knn.index"
    my_index = faiss.read_index(index_path)
    distances, indices = my_index.search(query_vector, k)

    dict = {}
    for i, (dist, indice) in enumerate(zip(distances[0], indices[0])):
        key_arr = []
        # Load the file names corresponding to the vector numbers
        # video_embeddings_folder = "/raid/datasets/msr-vtt/Test_scene_embeddings_"+suffix
        file_names = glob.glob(embedding_pa + "/*.npy")
        file_names = sorted(file_names)
        file_name = file_names[indice].split("/")[-1]
        key_arr.append(file_name)
        key_arr.append(dist)
        dict[i + 1] = key_arr
    return dict


def find_paths(
    model: str,
    index_path: str = "/raid/datasets/msr-vtt/index_folder_",
    embedding_path: str = "/raid/datasets/msr-vtt/scene_embeddings_",
    path: str = "/raid/datasets/msr-vtt/all_scene_mp4/",
):
    index_path += model[-2:]
    embedding_path += model[-2:]
    # all_vids
    return index_path, embedding_path, path


def retrieval(text, model):
    txt = find_text_embedding(text, model)

    ind_pa, embedding_pa, path = find_paths(model)

    arr = find_sim(txt, ind_pa, embedding_pa)
    ret_arr = []
    for i in range(len(arr)):
        x = arr[i + 1][0]
        # print(x)
        # to show all video
        s = x.find("-")

        e = x.find(".")
        vid_name = x[s + 1 : e]
        ret_arr.append(vid_name)
        path_i = path + vid_name + ".mp4"
        ret_arr.append(path_i)
    return ret_arr


def main():
    with gr.Interface(
        fn=retrieval,
        inputs=[
            gr.inputs.Textbox(label="Text", lines=2, placeholder="Text Here..."),
            gr.inputs.Dropdown(
                label="Model",
                choices=[
                    "microsoft/xclip-base-patch32",
                    "microsoft/xclip-base-patch16",
                    "microsoft/xclip-large-patch14",
                ],
            ),
        ],
        outputs=[
            gr.outputs.Textbox(label="Output 1"),
            gr.outputs.Video(label="Output 1"),
            gr.outputs.Textbox(label="Output 2"),
            gr.outputs.Video(label="Output 2"),
            gr.outputs.Textbox(label="Output 3"),
            gr.outputs.Video(label="Output 3"),
            gr.outputs.Textbox(label="Output 4"),
            gr.outputs.Video(label="Output 4"),
        ],
    ) as iface:
        iface.launch()


if __name__ == "__main__":
    fire.Fire(main)
