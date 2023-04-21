from autofaiss import build_index

embeddings_path = "/raid/datasets/msr-vtt/scene_embeddings_14"
index_folder_path = "/raid/datasets/msr-vtt/index_folder_14"

build_index(embeddings=embeddings_path, index_path= index_folder_path+"/Test_knn.index",
            index_infos_path=index_folder_path+"/index_infos.json", max_index_memory_usage="2G",
            current_memory_available="4G")
