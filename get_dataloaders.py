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

def stip_collate(batch):
  ''' Expected shape of 'frames': (num_cameras, num_channels, num_frames, width, height)'''
  vids = []
  for each in batch:
    vids += each['video'],
  return {'video': torch.stack(vids), 'frames': torch.stack([each['frames'] for each in batch], 0)}

def nuscenes_collate(batch):
  ''' Expected shape of 'frames': (num_cameras, num_channels, num_frames, width, height)'''
  return {'video': [], 'frames': torch.stack([each['frames'] for each in batch], 0)}

def get_data_loader(opt):
  if opt.dset_name.lower() == 'stip':
    print('Building STIPImgDataset...')
    dset = STIPImgDataset()
    print('Built STIPImgDataset.')
    collate_fn = stip_collate
  elif opt.dset_name.lower() == 'nuscenesmini':
    print('Building Mini NuScenesImgDataset...')
    dset = NuScenesImgDataset(name="v1.0-mini")
    print('Built Mini NuscenesImgDataset.')
    collate_fn = nuscenes_collate
  elif opt.dset_name.lower() == 'nuscenes':
    print('Building NuScenesImgDataset...')
    dset = NuScenesImgDataset()
    print('Built NuscenesImgDataset.')
    collate_fn = nuscenes_collate
  else:
    raise NotImplementedError('Sorry, we currently only support STIP and NuScenes.')
    
  train_size = int(0.8 * len(dset))
  test_size = len(dset) - train_size
  train_dset, test_dset = torch.utils.data.random_split(dset, [train_size, test_size])

  return data.DataLoader(train_dset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_workers,
    pin_memory=True,
    collate_fn=collate_fn,),
    data.DataLoader(test_dset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_workers,
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
