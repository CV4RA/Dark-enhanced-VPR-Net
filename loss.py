# models/loss.py
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        super(TripletLoss, self).__init__()
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=p)
    
    def forward(self, anchor, positive, negative):
        return self.triplet_loss_fn(anchor, positive, negative)

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce_loss_fn = nn.BCELoss()
    
    def forward(self, real_output, fake_output):
        real_loss = self.bce_loss_fn(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss_fn(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2

