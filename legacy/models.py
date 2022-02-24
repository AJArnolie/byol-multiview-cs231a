import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Projector, self).__init__()
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(output_dims)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(input_dims, 4096) 
        self.lin2 = nn.Linear(4096, output_dims) 

    def forward(self, x):
        x = self.relu(self.bn1(self.lin1(x)))
        return self.bn2(self.lin2(x))


class Predictor(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Predictor, self).__init__()
        self.bn1 = nn.BatchNorm1d(4096)
        self.lin1 = nn.Linear(input_dims, 4096) 
        self.lin2 = nn.Linear(4096, output_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.lin1(x)))
        return self.lin2(x)


class BYOLMultiView(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(BYOLMultiView, self).__init__()
        self.single = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        self.multi1 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.multi2 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.multi3 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.loss = Loss(input_dims, output_dims)

    def forward(self, s, m1, m2, m3):
        print("Before Model Calls", torch.cuda.memory_allocated())
        return self.single(s), self.multi1(m1), self.multi2(m2), self.multi3(m3)

    def get_optimizer(self):
        backbone, predictor = [], []
        for name, param in self.named_parameters():
            if "predictor" in name: predictor.append(param)
            else: backbone.append(param)

        opt = torch.optim.Adam([{"params": backbone, "lr": 0.001}, {"params": predictor, "lr": 0.01},])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode="min", factor=0.3, patience=3, verbose=True)
        return {"optimizer": opt, "lr_scheduler": scheduler}


class Loss(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Loss, self).__init__()
        self.projector = Projector(input_dims, output_dims)
        self.predictor = Predictor(output_dims, output_dims)
        self.loss = nn.MSELoss()  
  
    def forward(self, rep_s, rep_m1, rep_m2, rep_m3):     
        # Get representations for each single and multi-view video
        # Project embeddings (z)
        print("Before Projections", torch.cuda.memory_allocated())
        proj_s = self.projector(rep_s)
        proj_m1 = self.projector(rep_m1)
        proj_m2 = self.projector(rep_m2)
        proj_m3 = self.projector(rep_m3)
        print("After Projections", torch.cuda.memory_allocated())
        # Perform predictions (h)
        pred_s = self.predictor(proj_s)
        pred_m1 = self.predictor(proj_m1)
        pred_m2 = self.predictor(proj_m2)
        pred_m3 = self.predictor(proj_m3)
        print("After Predictions", torch.cuda.memory_allocated())
        # Get loss between three pairs
        L_s_m1 = self.loss(pred_s, proj_m1.detach())
        L_m1_s = self.loss(pred_m1, proj_s.detach())
        
        L_s_m2 = self.loss(pred_s, proj_m2.detach())
        L_m2_s = self.loss(pred_m2, proj_s.detach())

        L_s_m3 = self.loss(pred_s, proj_m3.detach())
        L_m3_s = self.loss(pred_m3, proj_s.detach())
        print("Loss", torch.cuda.memory_allocated())
        # Sum terms to get overall loss
        return L_s_m1 + L_m1_s + L_s_m2 + L_m2_s + L_s_m3 + L_m3_s
