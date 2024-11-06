import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from block import ResidualBlock
from attention import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from attention import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar

class ResidualAttentionEncoder(nn.Module):
    # for input size 32
    def __init__(self):
        super(ResidualAttentionEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        # self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(1024,128)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    x = torch.rand(32, 3, 32, 32)
    xc = x.detach().clone()
    net = ResidualAttentionEncoder()
    print(net(x).shape)
    print(torch.equal(x, xc))
          
    
