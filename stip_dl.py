import os
import numpy as np
import cv2
import random
random.seed(2019)
import torch
import torch.utils.data as data
​
class STIPImgDataset(data.Dataset):
  def __init__(self):
    self.fps = 2                    # Frame Rate of sampled clips    
    self.sample_interval = 300      # Interval at which a new set of frames is sampled
    self.frame_count = 8            # Number of frames per set of clips
    self.offset_update_rate = 500   # Interval at which offsets are updated
    
    self.data_path = '/vision/u/ajarno/STIP_dataset'   # Location at which frames and timestamp data are stored
    self.left_img_path = self.data_path + '/{:s}/L_camera'   
    self.right_img_path = self.data_path + '/{:s}/R_camera'
    self.center_img_path = self.data_path + '/{:s}/C_camera'
    self.img_path = '/{:06d}.jpg'   # (Frame ID)
    
    self.data = self.generate_frame_correspondences(self.fps, self.frame_count)
​
  def generate_frame_correspondences(self, fps, frame_count):
    ''' Returns list of tuples of format (video, [[left frame IDs], [center frame IDs], [right frame IDs]])
    '''
    print("STIPImgDataset Loading...")
    ret = []
    step = 20//fps
    invalid_count = 0
    for video in sorted(os.listdir(self.data_path))[:18]:
      # Loads in timestamp sync data to use to update offsets
      timestamps = np.genfromtxt(os.path.join(self.data_path, video, "timestamp_sync.txt"), delimiter=' ')
      ts_range = [int(timestamps[0, 2]), int(timestamps[-1, 2])]
      offset = [0, 0] # Left and Right Frame ID offsets in regard to the Center Frame ID
      data = []
​
      # Aggregates corresponding L, C, and R frame IDs into a list
      l = sorted(os.listdir(self.center_img_path.format(video)))
      llen = len(l)
      for num, frame in enumerate(l):
        f_number = int(frame.split('.')[0])
        # Periodically readjust offsets based on timestamp_sync data
        if f_number % self.offset_update_rate == 0 and ts_range[0] < f_number < ts_range[1]:
            if f_number in list(timestamps[:,2]):
                _, _, L, C, R = timestamps[list(timestamps[:,2]).index(f_number)]
                offset = [int(L - C), int(R - C)]
        
        # Only check ends of videos for invalid frame sets
        if num + 100 <= llen:
            data += [[f_number + offset[0], f_number, f_number + offset[1]]]
        elif os.path.exists(self.left_img_path.format(video) + self.img_path.format(f_number + offset[0])) and\
           os.path.exists(self.center_img_path.format(video) + self.img_path.format(f_number)) and\
           os.path.exists(self.right_img_path.format(video) + self.img_path.format(f_number + offset[1])):
            data += [[f_number + offset[0], f_number, f_number + offset[1]]]
        else:
            invalid_count += 1    
      
      # Samples aggregated list based on desired fps, frame count, and sample interval
      data = np.array(data)
      for i in range(0, data.shape[0] - (step * frame_count), self.sample_interval):
        ret += [(video, data[i:i + step * frame_count: step].T)] 
    print(str(invalid_count), "invalid frame sets found and removed.")
    print("STIPImgDataset loaded! (", len(ret), " clips loaded )")
    return ret
​
  def __len__(self):
    return len(self.data)
​
  def __getitem__(self, idx):
    ''' Returns dict containing name of video and tensor of expected shape: (num_cameras, num_channels, num_frames, W, H)
        Current Settings: (3, 32, 3, 224, 224)
    '''
    vid, fids = self.data[idx]
​
    # Loads in necessary frames based on frame IDs provided by self.data
    img_paths = [[self.left_img_path.format(vid) + self.img_path.format(fid) for fid in fids[0]],
                 [self.center_img_path.format(vid) + self.img_path.format(fid) for fid in fids[1]],
                 [self.right_img_path.format(vid) + self.img_path.format(fid) for fid in fids[2]]]
​
    cameras = []
    for camera in img_paths:
      frames = []
      for path in camera:
        img = cv2.resize(cv2.imread(path), (256, 256))
        frames += [img.transpose((2,0,1))]
      cameras += [torch.tensor(frames).permute((1, 0, 2, 3))]
    return {'L': cameras[0], 'C': cameras[1], 'R': cameras[2]}
