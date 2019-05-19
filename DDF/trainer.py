import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from ignite.engine import Engine

EPS = 1e-12


class Trainer():
    def __init__(self, model, optimizer, print_loss_every=50, record_loss_every=5,
                 use_cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.model.cuda()
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every

        # Initialize attributes
        self.num_steps = 0
        self.batch_size = None
        self.losses = {'loss': [],
                       'recon_loss': []}


    def train(self, data_loader, epochs=10):

        self.batch_size = data_loader.batch_size
        self.model.train()

        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(data_loader)
            print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1,
                                                          self.batch_size * self.model.num_pixels * mean_epoch_loss))


    def _train_epoch(self, data_loader):
        epoch_loss = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
                               # self.print_loss_every
        for batch_idx, (data, label) in enumerate(data_loader):
            iter_loss = self._train_iteration(data)
            epoch_loss += iter_loss
            print_every_loss += iter_loss
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                                  len(data_loader.dataset),
                                                  self.model.num_pixels * mean_loss))
                print_every_loss = 0.
        # Return mean epoch loss
        return epoch_loss / len(data_loader.dataset)

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, H, W)
        """
        self.num_steps += 1
        if self.use_cuda:
            data = data.cuda()

        self.optimizer.zero_grad()
        recon_batch, ddf = self.model(data)
        ddf = ddf**2
        ddf_reg = torch.mean(torch.sum(ddf, dim=1))
        loss = self._loss_function(data, recon_batch)#+ddf_reg/50
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss

    def _loss_function(self, data, recon_data):
        # Reconstruction loss is pixel wise cross-entropy
        recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels),
                                            data.view(-1, self.model.num_pixels))
        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels

        total_loss = recon_loss

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss / self.model.num_pixels
