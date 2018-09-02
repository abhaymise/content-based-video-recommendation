from pathlib import Path
import pandas as  pd
import numpy as np
from collections import OrderedDict

def get_file_encoding(path,filename=None,encoding_name='c3d',ext="npy"):
    files = (Path(path) / encoding_name).glob("*."+ext)
    return files

def get_vid_2d_encod():
    
    vid_cat = pd.read_csv('video-category.csv')

    mapping = OrderedDict(zip(vid_cat.iloc[:,0],vid_cat.iloc[:,1]))

    three_d_encode = sorted(get_file_encoding(Path.home() / '.cbvr'))
    
    all_ = pd.DataFrame()

    total_embeds= 0
    for val in three_d_encode:
        num = np.load(val)
        vid_id = val.stem.split("_")[0]
        feat_cols = [ 'pixel'+str(i) for i in range(num.shape[1]) ] 
        feat_data = pd.DataFrame(num,columns=feat_cols)
        feat_data['video_id'] = pd.Series(np.tile(int(vid_id),len(num)))
        feat_data['label'] = pd.Series(np.tile(mapping[int(vid_id)],len(num)))
        all_ = all_.append(feat_data,ignore_index=True)
        total_embeds+=len(num)
    
    assert len(all_) == total_embeds , "error in reading embeddings"
    
    return all_

