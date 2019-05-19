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
                       'recon_loss': [],
                       'kl_loss': []}


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
        recon_batch, latent_dist = self.model(data)
        loss = self._loss_function(data, recon_batch, latent_dist)
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss

    def _loss_function(self, data, recon_data, latent_dist):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy
        recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels),
                                            data.view(-1, self.model.num_pixels))
        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels

        mean, logvar = latent_dist
        kl_loss = self._kl_normal_loss(mean, logvar)

        # Calculate total loss
        total_loss = recon_loss + kl_loss

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss / self.model.num_pixels

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss'].append(kl_loss.item())
        return kl_loss
