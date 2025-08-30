import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from ScanWise.datasets.oasis_dataset import OASISDataset
from models.patchnet import PatchNet

class Trainer:
    def __init__(self, config):
        self.dataset = OASISDataset(config["dataset"]["path"])
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=config["dataloader"]["batch_size"],
                                     shuffle=True)

        self.model = PatchNet().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"])
        self.criterion = nn.MSELoss()
        self.epochs = config["training"]["epochs"]

    def run(self):
        for epoch in range(self.epochs):
            for imgs, targets in self.dataloader:
                imgs, targets = imgs.cuda(), targets.cuda()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {loss.item():.4f}")

