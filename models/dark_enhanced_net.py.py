# models/dark_enhanced_net.py
import torch
import torch.nn as nn
from models.res_em import ResEM_Generator, ResEM_Discriminator
from models.dsp_former import DSPFormer

class DarkEnhancedNet(nn.Module):
    def __init__(self, num_clusters=64):
        super(DarkEnhancedNet, self).__init__()
        self.generator = ResEM_Generator()
        self.discriminator = ResEM_Discriminator()
        self.dspformer = DSPFormer(num_clusters=num_clusters)

    def forward(self, x):
        enhanced_img = self.generator(x)
        global_descriptor = self.dspformer(enhanced_img)
        disc_result = self.discriminator(enhanced_img)
        return global_descriptor, disc_result

if __name__ == "__main__":
    model = DarkEnhancedNet(num_clusters=64).cuda()
    input_img = torch.randn(1, 3, 224, 224).cuda()
    global_descriptor, disc_result = model(input_img)
    print("Global Descriptor Shape:", global_descriptor.shape)
    print("Discriminator Output:", disc_result)
