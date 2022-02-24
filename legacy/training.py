import torch
import tqdm
import torch.nn as nn
from dataloader import STIPImgDataset
from models import BYOLMultiView, Loss

def train():
    EPOCHS = 1
    OUTPUT_DIR = "/vision/u/ajarno/cs231a/output"
    print(torch.cuda.memory_allocated())
    train_dataloader = get_data_loader("stip", batch_size=2, shuffle=True)
    val_dataloader = get_data_loader("stip", batch_size=2, shuffle=False)
    device = "cuda:0"
    print(torch.cuda.memory_allocated())
    model = BYOLMultiView(400, 400).to(device)
    loss = model.loss.to(device)
    optimizer = model.get_optimizer()["optimizer"]
    best_val_loss = float('inf')
    print(torch.cuda.memory_allocated())
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for data in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
            m1 = data['L'].type(torch.FloatTensor).to(device)
            m2 = data['C'].type(torch.FloatTensor).to(device)
            m3 = data['R'].type(torch.FloatTensor).to(device)
            print("Load Data", torch.cuda.memory_allocated())
            train_loss = loss(*model(m2, m1, m2, m3))
            print("Loss", torch.cuda.memory_allocated())
            train_loss.backward()
            print("Backward", torch.cuda.memory_allocated())
            optimizer.step()
            train_losses.append(train_loss.item())
            print("Optimizer", torch.cuda.memory_allocated())
            torch.cuda.empty_cache()
        train_loss_avg = np.mean(train_losses)
        scheduler.step()

        model.eval()
        val_losses = []
        for data in enumerate(val_dataloader):
            m1 = data['L'].type(torch.FloatTensor).to(device)
            m2 = data['C'].type(torch.FloatTensor).to(device)
            m3 = data['R'].type(torch.FloatTensor).to(device)
            val_loss = loss(*model(m2, m1, m2, m3))
            val_losses.append(val_loss.item())

        #torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_last.pth'))
        val_loss_avg = np.mean(val_losses)
        print(f'epoch {epoch:03d}:\t'
              f'train loss: {train_loss_avg:.4f}\t'
              f'val loss: {val_loss_avg:.4f}')
        if best_val_loss > val_loss_avg:
            #torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_best.pth'))
            best_val_loss = val_loss_avg

def collate(batch):
  ''' Expected shape of L,R,C: (num_channels, num_frames, width, height)'''
  return {'L': torch.stack([each['L'] for each in batch], 0),
          'C': torch.stack([each['C'] for each in batch], 0),
          'R': torch.stack([each['R'] for each in batch], 0)}

def get_data_loader(name, batch_size=16, shuffle=True, num_workers=0):
  if name.lower() == 'stip':
    dset = STIPImgDataset()
    print('Built STIPImgDataset')
    collate_fn = collate
  else:
    raise NotImplementedError('Sorry, we currently only support STIP.')
  return torch.utils.data.DataLoader(dset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate)

if __name__ == '__main__':
    train()

