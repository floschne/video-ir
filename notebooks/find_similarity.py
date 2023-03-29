import numpy as np
from autofaiss import build_index
import faiss
import glob

# loadinf text embedding
txt_array = np.load('/srv/home/ahmadi/gitrepos/video-ir/text_(news).npy') 

build_index(embeddings="/srv/home/ahmadi/gitrepos/video-ir/vid_embedding_np", index_path="/srv/home/ahmadi/gitrepos/video-ir/index_folder/knn.index",
            index_infos_path="/srv/home/ahmadi/gitrepos/video-ir/index_folder/index_infos.json", max_index_memory_usage="1G",
            current_memory_available="3G")

my_index = faiss.read_index("/srv/home/ahmadi/gitrepos/video-ir/index_folder/knn.index")

# Load the file names corresponding to the vector numbers
file_names = glob.glob('/srv/home/ahmadi/gitrepos/video-ir/vid_embedding_np/*.npy')
file_names = sorted(file_names)

query_vector = txt_array
k = 5
distances, indices = my_index.search(query_vector, k)

print(f"Top {k} elements in the dataset for max inner product search:")
for i, (dist, indice) in enumerate(zip(distances[0], indices[0])):
  file_name = file_names[indice].split('/')[-1] # Subtract 1 from indice to convert to 0-based index
  print(f"{i+1}: Vector number {indice:4} ({file_name}) with distance {dist}")