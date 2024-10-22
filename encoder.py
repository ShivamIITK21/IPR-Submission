import torch.nn as nn
import torch
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    def __init__(self, image_dim, latent_dim, pretrained_weights=True):
        super().__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        if(pretrained_weights):
            net = resnet18(num_classes=self.latent_dim)
        else:
            net = resnet18(weigts=None, num_classes=self.latent_dim)

        self.encoder = []
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                self.encoder.append(nn.Flatten(1))
                self.encoder.append(module)
            else:
                if name == 'conv1':
                    module = nn.Conv2d(module.in_channels, module.out_channels,
                                       kernel_size=3, stride=1, padding=1, bias=False)
                    if isinstance(module, nn.MaxPool2d):
                        continue
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        return self.encoder(x) 

if __name__ == "__main__":
    x = torch.rand(3, 3, 64, 64)
    net = ResNetEncoder(64, 128, False)
    print(net(x).shape)




    

