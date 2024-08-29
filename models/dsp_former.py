# models/dsp_former.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

def __init__(self, num_clusters=64):
        super(DSPFormer, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  
        
        self.downsample = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.netvlad = NetVLAD(num_clusters=num_clusters)

    def forward(self, x):
        x = self.downsample(x)
        features = self.vit(x)
        features = self.upsample(features)
        global_descriptor = self.netvlad(features)
        return global_descriptor


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=768):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, dim))
        self.fc = nn.Linear(num_clusters * dim, num_clusters)

    def forward(self, x):
        N, C, H, W = x.size()
        x_flatten = x.view(N, C, H * W)  # Flatten the spatial dimensions
        x_flatten = x_flatten.permute(0, 2, 1)  # N x HW x C
        x_expanded = x_flatten.unsqueeze(1)  # N x 1 x HW x C
        cluster_centers_expanded = self.cluster_centers.unsqueeze(0).unsqueeze(2)  # 1 x K x 1 x C
        distances = (x_expanded - cluster_centers_expanded).pow(2).sum(3)  # N x K x HW
        soft_assign = F.softmax(-distances, dim=1)  # N x K x HW
        
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for i in range(self.num_clusters):
            residual = x_flatten - self.cluster_centers[i:i + 1, :]
            residual *= soft_assign[:, i:i + 1, :].expand_as(residual)
            vlad[:, i:i + 1, :] = residual.sum(dim=1)

        vlad = vlad.view(x.size(0), -1)  # Flatten to N x (K*C)
        vlad = self.fc(vlad)  # Add a fully connected layer for classification
        return vlad
