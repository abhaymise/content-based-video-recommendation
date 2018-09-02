
## reading videos as c3d vectors and saving them on disk
import sys
sys.path.insert(0,'/home/deep-vision/Documents/User/abhay/video_spatio_temporal_works/c3d')

import skvideo.io
from keras.models import Model
from c3d import C3D
from sports1M_utils import preprocess_input, decode_predictions
import numpy as np
from pathlib import Path

batch_size = 16

model=None
def load_model():
    global model
    base_model = C3D(weights='sports1M')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

load_model()
print("[INFO] C3D model loaded ...")

def get_features(vid):
    x = preprocess_input(vid)
    features = model.predict(x)
    return features

def batch(video_len,batch_size):
    for idx in range(0,video_len,batch_size):
        yield range(idx,min(idx+batch_size,video_len))
        
def get_3d_feature(vid_path,batch_size=16):
    name = vid_path
    vid = skvideo.io.vread(name)
    print('video stats ',vid.shape)
    bat = batch(vid.shape[0],batch_size)
    batch_indexes = list(bat)
    print('total temporal vectors ',len(batch_indexes))
    arr = np.empty((0,4096))
    for val in batch_indexes:
        arr=np.append(arr,get_features(vid[val]),axis=0) 
    return arr

def save_file_encoding(arr,path,filename='default',encoding_name='c3d'):
    h,w = arr.shape
    file_dir = Path(path) / encoding_name 
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = str((file_dir / "{}_{}_{}.npy".format(filename,h,w)).absolute())
    print(file_path)
    np.save(file_path,arr)

def get_file_encoding(path,filename=None,encoding_name='c3d',ext="npy"):
    files = (Path(path) / encoding_name).glob("*."+ext)
    return files

def save_all_encoding(all_paths):
    for idx,vid in enumerate(all_paths):
        name = str(vid.absolute())
        vid_id = vid.stem
        print("processing video ",vid_id)
        arr = get_3d_feature(name,batch_size)
        target_path =  (Path.home() / '.cbvr' )
        save_file_encoding(arr, target_path,filename=vid_id)
        print("Loading" + "." * idx)
        sys.stdout.write("\033[F")
        print("[INFO] {} saved to {} ".format(vid_id,str(target_path)))

