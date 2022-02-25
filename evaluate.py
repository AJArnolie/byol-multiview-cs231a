import numpy as np
import torch
import tqdm
import torch.nn as nn
from get_dataloaders import get_data_loader
from models import BYOLMultiView, Loss

def evaluate():
    EPOCHS = 30
    OUTPUT_DIR = "/vision/u/ajarno/cs231a/output"
    
    train_dataloader, val_dataloader = get_data_loader("NuScenes", batch_size=4, num_workers=4)
    device = "cuda:0"
    model = BYOLMultiView(400, 400).to(device)
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for data in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
            m1 = data['L'].type(torch.FloatTensor).to(device)
            m2 = data['C'].type(torch.FloatTensor).to(device)
            m3 = data['R'].type(torch.FloatTensor).to(device)
            #m1 = torch.tensor(data['L'], dtype=torch.float, device=device)
            #m2 = torch.tensor(data['C'], dtype=torch.float, device=device)
            #m3 = torch.tensor(data['R'], dtype=torch.float, device=device)
            train_loss = loss(*model(m2, m1, m3))
            m1, m2, m3 = None, None, None
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            train_loss = None
            torch.cuda.empty_cache()
        train_loss_avg = np.mean(train_losses)
        #scheduler.step()

        model.eval()
        val_losses = []
        for i, data in enumerate(val_dataloader):
            m1 = data['L'].type(torch.FloatTensor).to(device)
            m2 = data['C'].type(torch.FloatTensor).to(device)
            m3 = data['R'].type(torch.FloatTensor).to(device)
            #m1 = torch.tensor(data['L'], dtype=torch.float, device=device)
            #m2 = torch.tensor(data['C'], dtype=torch.float, device=device)
            #m3 = torch.tensor(data['R'], dtype=torch.float, device=device) 
            val_loss = loss(*model(m2, m1, m3))
            m1, m2, m3 = None, None, None
            val_losses.append(val_loss.item())
            val_loss = None
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_last.pth'))
        val_loss_avg = np.mean(val_losses)
        print(f'epoch {epoch:03d}:\t'
              f'train loss: {train_loss_avg:.4f}\t'
              f'val loss: {val_loss_avg:.4f}')
        f.write(f'epoch {epoch:03d}:\t'
              f'train loss: {train_loss_avg:.4f}\t'
              f'val loss: {val_loss_avg:.4f}')
        if best_val_loss > val_loss_avg:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_best.pth'))
            best_val_loss = val_loss_avg


def collate(batch):
  ''' Expected shape of L,R,C: (num_channels, num_frames, width, height)'''
  return {'L': torch.stack([each['L'] for each in batch], 0),
          'C': torch.stack([each['C'] for each in batch], 0),
          'R': torch.stack([each['R'] for each in batch], 0)}

def get_data_loaders(name, batch_size=16, shuffle=True, num_workers=0):
  if name.lower() == 'stip':
    dset = STIPImgDataset()
    train_size = int(0.8 * len(dset))
    test_size = len(dset) - train_size
    train_dset, test_dset = torch.utils.data.random_split(dset, [train_size, test_size])
    print('Built STIPImgDataset')
    collate_fn = collate
  else:
    raise NotImplementedError('Sorry, we currently only support STIP.')
  return torch.utils.data.DataLoader(train_dset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate), torch.utils.data.DataLoader(test_dset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate)

if __name__ == '__main__':
    evaluate()

