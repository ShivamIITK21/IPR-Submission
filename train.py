from encoder import ResNetEncoder
from denoiser import UNet_decoder
from soda import SODA
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

DEVICE="cuda"
IMAGE_DIM = 32
LATENT_DIM = 128
LR = 1.0e-4
BATCH_SIZE = 32
CHECKPT_DIR = "./checkpoints"
IMAGES_DIR = "./images"
DATA_ROOT = "./data/cifar10"
TB_DIR = "./tensorboard_logs"
EPOCHS=800
LOAD_EPOCH = 0 

class SourceTarget(Dataset):
    def __init__(self):
        super().__init__()
        source_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                    ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.source_data = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=source_transform, download=True)
        self.target_data = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=target_transform)

    def __getitem__(self, idx):
        return self.source_data.__getitem__(idx), self.target_data.__getitem__(idx)
    
    def __len__(self):
        return self.source_data.__len__()



if __name__ == "__main__":
    os.makedirs(CHECKPT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=TB_DIR)


    data = SourceTarget()
    dataloader = DataLoader(dataset=data, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

    encoder = ResNetEncoder(image_dim=IMAGE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    denoiser = UNet_decoder(image_shape=(3, IMAGE_DIM, IMAGE_DIM), z_channels=LATENT_DIM).to(DEVICE)
    soda = SODA(encoder, denoiser, betas=[1.0e-4, 0.02], n_T=1000, drop_prob=0.1, device=DEVICE).to(DEVICE)

    opt = optim.AdamW([{'params': soda.encoder.parameters(), 'lr': LR * 2}, {'params': soda.decoder.parameters(), 'lr': LR}], weight_decay=0.05, betas=[0.9, 0.95])

    if LOAD_EPOCH != 0:
        checkpoint = torch.load(os.path.join(CHECKPT_DIR, f"model_{LOAD_EPOCH}.pth"), map_location=DEVICE)
        soda.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
            
        print("Loaded checkpoint")

    for ep in range(LOAD_EPOCH, EPOCHS):
        ep_loss = 0
        for batch_idx, (source, target) in enumerate(tqdm(dataloader)):
            source = source[0].to(DEVICE)
            target = target[0].to(DEVICE)
            opt.zero_grad()
            loss = soda(source, target)
            ep_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=soda.parameters(), max_norm=1)
            opt.step()

            if batch_idx == 0:
                with torch.no_grad():
                    z = soda.encoder(source[:16])
                    x_gen = soda.ddim_sample(16, target.shape[1:], z)
                    x_source = source[:16]
                    x_real = target[:16]
                    x_all = torch.cat([x_gen.cpu(), x_real.cpu(), x_source.cpu()])
                    save_path = os.path.join(IMAGES_DIR, f"image_ep_{ep}.png")
                    grid = make_grid(x_all, nrow=8)
                    save_image(grid, save_path)
                    writer.add_image(f"Reconstruction/Orignal/EncoderIn_{ep}", grid, ep)

        checkpoint = {
            'model': soda.state_dict(),
            'opt': opt.state_dict(),
        }
        save_path = os.path.join(CHECKPT_DIR, f"model_{ep}.pth")
        torch.save(checkpoint, save_path)
        try:
            if((ep-2)%100 != 0):
                os.remove(os.path.join(CHECKPT_DIR, f"model_{ep-2}.pth"))
        except OSError:
            pass

        writer.add_scalar("Loss/Batch", ep_loss, ep)
        print(f"Loss as {ep} epoch: {ep_loss}")




