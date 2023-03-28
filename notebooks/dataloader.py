import os
import torch
import torch.utils.data as data
# import torchvision.transforms as transforms
import torchvision.datasets.video_utils
import glob
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np
import math

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir):
        self.video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video, _, info = torchvision.datasets.video_utils.read_video(video_path, pts_unit="sec")
        return video

    def __len__(self):
        return len(self.video_paths)

        return video

def collate_fn(batch):
    # Sort the batch by video length
    batch.sort(key=lambda x: x.shape[0], reverse=True)

    # Create a list of tensors representing each video in the batch
    videos = []
    for video in batch:
        if isinstance(video, np.ndarray):
            videos.append(torch.from_numpy(video))
        else:
            videos.append(video)

    # Pack the videos into a single PackedSequence
    packed_videos = pack_sequence(videos)

    # Pad the packed sequence
    padded_videos, _ = pad_packed_sequence(packed_videos, batch_first=True)

    return padded_videos




dataset = VideoDataset(video_dir="/raid/datasets/msr-vtt/Test2")
first_data = dataset[0]
# print(first_data)

dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

# print(total_samples, n_iterations)

for i, input in enumerate(dataloader):
    if (i+1) % 2 == 0:
        print(f' epoch = 1 , step {i+1}/{n_iterations}, inputs {input.shape}')



