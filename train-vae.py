from utils.dataloaders import get_mnist_dataloaders
from VAE.model import VAE
from torch import optim
from VAE.trainer import Trainer
import torch

train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
model = VAE(img_size=(1, 32, 32), latent_dim=36, use_cuda=True)
print(model)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Build a trainer
trainer = Trainer(model, optimizer, use_cuda=True)
trainer.train(train_loader, epochs=100)

torch.save(model.state_dict(), "models/vae_mnist.pt")