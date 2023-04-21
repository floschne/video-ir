import av
import numpy as np
import os
from tqdm import tqdm
import fire

from transformers import AutoProcessor, AutoModel

np.random.seed(0)

import os


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def main_func(
    model_name: str = "microsoft/xclip-large-patch14",
    scene_dir_path: str = "/raid/datasets/msr-vtt/Test_scene_mp4",
    save_path: str = "/raid/datasets/msr-vtt/scene_embeddings_14",
    error_path: str = "/srv/home/ahmadi/gitrepos/video-ir/err_embedding_path.txt",
):
    er = open(error_path, "a+")
    er.seek(0)

    f = []
    for path in os.listdir(save_path):
        p = os.path.join(save_path, path)
        f.append(p)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda:0")
    # model = AutoModel.from_pretrained(model_name)

    for path in tqdm(os.listdir(scene_dir_path), desc="Iterate over each scene..."):
        path2 = os.path.join(scene_dir_path, path)
        z = save_path + "/np-" + path[:-4] + ".npy"

        if z not in f:
            try:
                container = av.open(path2)
                indices = sample_frame_indices(
                    clip_len=8,
                    frame_sample_rate=1,
                    seg_len=container.streams.video[0].frames,
                )

                video = read_video_pyav(container, indices)

                inputs = processor(videos=list(video), return_tensors="pt")
                inputs = inputs.to("cuda:0")

                video_features = model.get_video_features(**inputs)

                np_arr = video_features.cpu().detach().numpy()

                np.save(save_path + "/np-" + path[:-4] + ".npy", np_arr)

            except Exception as e:
                print(f"Error with file {path2}:\n\t{e}")
                er.seek(0)
                er.write(path2 + "\n")


if __name__ == "__main__":
    fire.Fire(main_func)
