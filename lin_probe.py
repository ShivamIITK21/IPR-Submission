import torch.nn as nn
from soda import SODA
from encoder import ResNetEncoder
from denoiser import UNet_decoder
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import DATA_ROOT

MODEL_PATH = "./checkpoints/model.pth"
DATA = "../data/cifar10"
LATENT_DIM = 128
NUM_CLASSES = 10
IMAGE_DIM = 32
DEVICE = "cuda"
BATCH_SIZE=256
EPOCHS = 15 
LR = 1e-3


class Simpleclassifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.norm = nn.BatchNorm1d(self.latent_dim, affine=False)
        self.lin = nn.Linear(self.latent_dim, self.num_classes)

    def forward(self, x, model):
        features = model.encoder(x)
        norm_f = self.norm(features)
        logits = self.lin(norm_f)
        return logits

if __name__ == "__main__":
    sc = Simpleclassifier(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    encoder = ResNetEncoder(image_dim=IMAGE_DIM,latent_dim=LATENT_DIM)
    denoiser = UNet_decoder(image_shape=(3, IMAGE_DIM, IMAGE_DIM), z_channels=LATENT_DIM).to(DEVICE)
    soda = SODA(encoder, denoiser, betas=[1.0e-4, 0.02], n_T=1000, drop_prob=0.1, device=DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    soda.load_state_dict(checkpoint["model"])
    print("Loaded Model")

    source_transform = transforms.Compose([
                transforms.RandomApply([
            transforms.RandomResizedCrop(IMAGE_DIM),
            transforms.RandomHorizontalFlip(),
        ], p=0.65),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    train_data = datasets.CIFAR10(root=DATA_ROOT,download=True, train=True, transform=source_transform) 
    test_data = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform)

    train_dl = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=4)
    test_dl = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(sc.parameters(), lr=LR)

    sc.train()
    for ep in range(0, EPOCHS):
        batch_loss = 0
        for batch_idx, (x, label) in enumerate(tqdm(train_dl)):
            x = x.to(DEVICE)
            label = label.to(DEVICE)

            logits = sc(x, soda)
            loss = loss_fn(logits, label)
            batch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        

    sc.eval()
    preds = []
    truths = []
    for batch_idx, (x, true_label) in enumerate(tqdm(test_dl)):
        x = x.to(DEVICE)
        true_label = true_label.to(DEVICE)
        logits = sc(x, soda)
        pred_class = logits.argmax(dim=-1)
        preds.append(pred_class)
        truths.append(true_label)
    preds = torch.cat(preds)
    truths = torch.cat(truths)
    print(len(preds))
    acc = (preds == truths).sum().item() / len(preds)
    print(f"Accuracy for linear probe : {acc}")

    top3_correct = 0
    total_samples = 0

    for batch_idx, (x, true_label) in enumerate(tqdm(test_dl)):
        x = x.to(DEVICE)
        true_label = true_label.to(DEVICE)
        logits = sc(x, soda)

        _, top3_preds = torch.topk(logits, 3, dim=1)

        correct_top3 = top3_preds.eq(true_label.view(-1, 1))

        top3_correct += correct_top3.sum().item()
        total_samples += true_label.size(0)

    top3_accuracy = top3_correct / total_samples

    print(f'Top 3 accuracy: {top3_accuracy}')
            
            



