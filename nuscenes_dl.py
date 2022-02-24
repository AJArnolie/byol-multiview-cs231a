import os
import numpy as np
import cv2
import random
import torch
import torch.utils.data as data
from nuscenes.nuscenes import NuScenes
​
class NuScenesImgDataset(data.Dataset):
  def __init__(self, name="v1.0-trainval"):
    self.fps = 2                    # Desired Frame Rate of sampled clips    
    self.sample_interval = 4        # Interval at which a new set of frames is sampled
    self.frame_count = 32           # Number of frames per set of clips
    
    self.data_path = '/vision/u/ajarno/NuScenes/nuScenes/'   # Location at which frames are stored
    self.nusc = NuScenes(version=name, dataroot=self.data_path, verbose=False)
    self.data = self.get_frame_correspondences(self.fps, self.frame_count)
​
  def get_frame_correspondences(self, fps, frame_count):
    ''' Returns list of tuples of format (video, [[left frame IDs], [center frame IDs], [right frame IDs]])
    '''
    ret = []
    step = 2//fps
    for scene in self.nusc.scene:
      #print(scene['description'])
      curr_sample = self.nusc.get('sample', scene['first_sample_token'])
      data = [[self.nusc.get('sample_data', curr_sample['data']['CAM_FRONT_LEFT'])['filename'], 
               self.nusc.get('sample_data', curr_sample['data']['CAM_FRONT'])['filename'],
               self.nusc.get('sample_data', curr_sample['data']['CAM_FRONT_RIGHT'])['filename']]]
      while curr_sample['next'] != "":
        curr_sample = self.nusc.get('sample', curr_sample['next'])
        data += [[self.nusc.get('sample_data', curr_sample['data']['CAM_FRONT_LEFT'])['filename'],
                  self.nusc.get('sample_data', curr_sample['data']['CAM_FRONT'])['filename'],
                  self.nusc.get('sample_data', curr_sample['data']['CAM_FRONT_RIGHT'])['filename']]]
      
      # Samples aggregated list based on desired fps, frame count, and sample interval
      data = np.array(data)
      for i in range(0, data.shape[0] - (step * frame_count), self.sample_interval):
        ret += [data[i:i + step * frame_count: step].T]
    print("NuScenes dataset loaded! (", len(ret), "clips loaded )") 
    return ret
​
  def __len__(self):
    return len(self.data)
​
  def __getitem__(self, idx):
    ''' Returns dict containing name of video and tensor of expected shape: (num_cameras, num_channels, num_frames, W, H)
        Current Settings: (3, 32, 3, 224, 224)
    '''
    fids = self.data[idx]
    img_paths = [fids[0], fids[1], fids[2]]
    cameras = []
    for camera in img_paths:
      frames = []
      for path in camera:
        img = cv2.resize(cv2.imread(self.data_path + path), (224, 224))
        frames += [img.transpose((2,0,1))]
      cameras += [frames]
    cameras = torch.tensor(cameras)
    return {'frames': cameras}
