import torch.nn as nn
import torch
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    def __init__(self, image_dim, latent_dim, pretrained_weights=True):
        super().__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        if(pretrained_weights):
            self.net = resnet18(num_classes=self.latent_dim)
        else:
            self.net = resnet18(weigts=None, num_classes=self.latent_dim)

    def forward(self, x):
        return self.net(x) 

if __name__ == "__main__":
    x = torch.rand(3, 3, 64, 64)
    net = ResNetEncoder(64, 128, False)
    print(net(x).shape)




    

