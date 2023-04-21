import os
import torch
import torch.utils.data as data
# import torchvision.transforms as transforms
import torchvision.datasets.video_utils
import glob
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np
import math
import make_scenes_files


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir):
        self.video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video, _, info = torchvision.datasets.video_utils.read_video(video_path, pts_unit="sec")
        return video, video_path

    def __len__(self):
        return len(self.video_paths)
    

def collate_fn(batch):
    # Sort the batch by video length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    # Extract the videos and their file paths from the tuples
    videos = [x[0] for x in batch]
    video_paths = [x[1] for x in batch]

    # Convert numpy arrays to PyTorch tensors
    videos = [torch.from_numpy(video) if isinstance(video, np.ndarray) else video for video in videos]

    # Pack the videos into a single PackedSequence
    packed_videos = pack_sequence(videos)

    # Pad the packed sequence
    padded_videos, _ = pad_packed_sequence(packed_videos, batch_first=True)

    return padded_videos, video_paths


p = "/raid/datasets/msr-vtt/TestVideo"

dataset = VideoDataset(video_dir=p)
# first_data = dataset[0]
# print(first_data)

dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)



# dataiter = iter(dataloader)
# data1 = dataiter.next()
# print(data1)

total_samples = len(dataset)
n_iterations = math.ceil(total_samples/64)

print(total_samples, n_iterations)

batch_size=64

for i, batch in enumerate(dataloader):
    print(batch[1])
    # make_scenes_files.scene_txt(batch[1], p)

