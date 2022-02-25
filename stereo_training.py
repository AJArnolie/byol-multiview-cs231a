import os
import numpy as np
import torch
import tqdm
import torch.nn as nn
from get_dataloaders import get_data_loader
from stereo_models import BYOLMultiView, Loss

def train():
    EPOCHS = 30
    OUTPUT_DIR = "/vision/u/ajarno/cs231a/output"
    
    train_dataloader, val_dataloader = get_data_loader("stip", batch_size=4, num_workers=4)
    device = "cuda:0"
    model = BYOLMultiView(400, 400).to(device)
    loss = model.loss.to(device)
    opt = model.get_optimizer()
    optimizer = opt["optimizer"]
    #scheduler = opt["lr_scheduler"]
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for data in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
            m1 = data['L'].type(torch.FloatTensor).to(device)
            m2 = data['C'].type(torch.FloatTensor).to(device)
            m3 = data['R'].type(torch.FloatTensor).to(device)
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
        if best_val_loss > val_loss_avg:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_best.pth'))
            best_val_loss = val_loss_avg

if __name__ == '__main__':
    train()

