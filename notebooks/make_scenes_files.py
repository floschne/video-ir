import os 

#this code, makes all .scene.txt files of directory bellow
os.chdir("/srv/home/ahmadi/gitrepos/video-ir/notebooks/TransNetV2/inference")

for path in os.listdir("/raid/datasets/msr-vtt/Test2"):
    path = os.path.join("/raid/datasets/msr-vtt/Test2", path)
    print(path)
    string = "python transnetv2.py "
    com = string + path 
    os.system(com)


    


