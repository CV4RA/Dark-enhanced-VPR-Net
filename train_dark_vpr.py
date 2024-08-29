# train/train_dark_vpr.py
import torch
from torch import optim
from models.dark_enhanced_net import DarkEnhancedNet
from models.losses import TripletLoss, AdversarialLoss  
from data.dark_place_loader import get_data_loaders


model = DarkEnhancedNet(num_clusters=64).cuda()


triplet_loss_fn = TripletLoss(margin=1.0, p=2).cuda()
adversarial_loss_fn = AdversarialLoss().cuda()


optimizer = optim.Adam(model.parameters(), lr=1e-4)


train_loader, val_loader = get_data_loaders()


for epoch in range(20):  
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        
        optimizer.zero_grad()
        
        anchor_embedding, _ = model(anchor)
        positive_embedding, _ = model(positive)
        negative_embedding, _ = model(negative)

        real_output = model.discriminator(anchor)
        fake_output = model.discriminator(model.generator(anchor).detach())
        
        triplet_loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        adversarial_loss = adversarial_loss_fn(real_output, fake_output)
        loss = triplet_loss + adversarial_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
