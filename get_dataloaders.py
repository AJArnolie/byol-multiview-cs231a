import torch
import random
import torch.utils.data as data
import argparse
try:
  from .stip_dl import STIPImgDataset
except:
  from stip_dl import STIPImgDataset
try:
  from .nuscenes_dl import NuScenesImgDataset
except:
  from nuscenes_dl import NuScenesImgDataset
 
def collate(batch):
  ''' Expected shape of L,R,C: (num_channels, num_frames, width, height)'''
  return {'L': torch.stack([each['L'] for each in batch], 0),
          'C': torch.stack([each['C'] for each in batch], 0),
          'R': torch.stack([each['R'] for each in batch], 0)}

def get_data_loader(name="STIP", batch_size=4, num_workers=4, pct_train=0.8):
  if name.lower() == 'stip':
    print('Building STIPImgDataset...')
    dset = STIPImgDataset()
    print('Built STIPImgDataset.')
    collate_fn = collate
  elif name.lower() == 'nuscenesmini':
    print('Building Mini NuScenesImgDataset...')
    dset = NuScenesImgDataset(name="v1.0-mini")
    print('Built Mini NuscenesImgDataset.')
    collate_fn = collate
  elif name.lower() == 'nuscenes':
    print('Building NuScenesImgDataset...')
    dset = NuScenesImgDataset()
    print('Built NuscenesImgDataset.')
    collate_fn = collate
  else:
    raise NotImplementedError('Sorry, we currently only support STIP and NuScenes.')
    
  train_size = int(pct_train * len(dset))
  test_size = len(dset) - train_size
  train_dset, test_dset = torch.utils.data.random_split(dset, [train_size, test_size])

  return data.DataLoader(train_dset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn,),
    data.DataLoader(test_dset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn,)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--is-train', type=bool, default=True)
  parser.add_argument('--dset-name', type=str, default='STIP')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--n-workers', type=int, default=0)
  opt = parser.parse_args()

  if True: # For testing purposes
    dloader = get_data_loader(opt)
    for i in range(1):
        n = random.randint(0, len(dloader.dataset) - 1) # Select random index
        print(dloader.dataset.__getitem__(n)['frames'].shape)
