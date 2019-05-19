import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np

EPS = 1e-12


class DDF(nn.Module):
    def __init__(self, img_size, latent_dim,  use_cuda=False):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        temperature : float
            Temperature for gumbel softmax distribution.

        use_cuda : bool
            If True moves model to GPU
        """
        super(DDF, self).__init__()
        self.use_cuda = use_cuda

        # Parameters
        self.img_size = img_size
        self.num_pixels = img_size[1] * img_size[2]
        self.hidden_dim = 10000
        self.hidden_dim2 = 256
        self.reshape = (64, 4, 4)  # Shape required to start transpose convs

        # Calculate dimensions of latent distribution
        self.latent_dim = latent_dim

        # Define encoder layers
        # Intial layer
        encoder_layers = [
            nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]
        # Add additional layer if (64, 64) images
        if self.img_size[1:] == (64, 64):
            encoder_layers += [
                nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]
        elif self.img_size[1:] == (32, 32):
            # (32, 32) images are supported but do not require an extra layer
            pass
        else:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        # Add final layers
        encoder_layers += [
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]

        # Define encoder
        self.img_to_features = nn.Sequential(*encoder_layers)

        # Map encoded features into a hidden vector which will be used to
        # encode parameters of the latent distribution
        self.features_to_hidden = nn.Sequential(
            nn.Linear(64 * 4 * 4, self.hidden_dim),
            nn.ReLU()
        )

        self.ddf_input = nn.Linear(self.hidden_dim, self.latent_dim)
        self.ddf_output = nn.Linear(self.latent_dim, self.hidden_dim2)
        torch.nn.init.xavier_uniform_(self.ddf_input.weight)
        torch.nn.init.xavier_uniform_(self.ddf_output.weight)

        for param in self.ddf_input.parameters():
            param.requires_grad = False
        # for param in self.ddf_output.parameters():
        #     param.requires_grad = False

        w = self.ddf_input.weight.numpy()
        b_input = self.ddf_input.bias.numpy()
        products = []
        for i in range(self.latent_dim):
            for j in range(i+1, self.latent_dim):
                w1 = w[i]
                w2 = w[j]
                prod = np.dot(w1, w2) + b_input[i]
                products.append(prod)
        shift = np.percentile(products, 99)
        biases_input = self.ddf_input.bias.numpy()
        biases_input -= shift
        # biases_output = self.ddf_output.bias.numpy()
        # biases_output -= shift

        # Map latent samples to features to be used by generative model
        self.latent_to_features = nn.Sequential(
            nn.ReLU(),
            self.ddf_output,
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, 64 * 4 * 4),
            nn.ReLU()
        )

        # Define decoder
        decoder_layers = []

        # Additional decoding layer for (64, 64) images
        if self.img_size[1:] == (64, 64):
            decoder_layers += [
                nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]

        decoder_layers += [
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.img_size[0], (4, 4), stride=2, padding=1),
            nn.Sigmoid()
        ]

        # Define decoder
        self.features_to_img = nn.Sequential(*decoder_layers)

    def decode(self, latent_sample):
        """
        Decodes sample from latent distribution into an image.

        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        features = self.latent_to_features(latent_sample)
        return self.features_to_img(features.view(-1, *self.reshape))

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (N, C, H, W)
        """
        batch_size = x.size()[0]

        # Encode image to hidden features
        torch.nn.init.xavier_uniform_(self.ddf_input.weight)

        features = self.img_to_features(x)
        hidden = self.features_to_hidden(features.view(batch_size, -1))

        ddf = self.ddf_input(hidden)

        return self.decode(ddf), ddf
