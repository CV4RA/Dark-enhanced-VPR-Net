# eval/eval_dark_vpr.py
import torch
from models.dark_enhanced_net import DarkEnhancedNet
from data.dark_place_loader import get_data_loaders

model = DarkEnhancedNet(num_clusters=64).cuda()
model.load_state_dict(torch.load('path_to_trained_model'))  

_, val_loader = get_data_loaders()

def evaluate():
    model.eval()
   
evaluate()
