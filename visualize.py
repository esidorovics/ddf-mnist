import torch
from torchvision.utils import make_grid, save_image
import numpy as np


class Visualize:
    def __init__(self, model, use_cuda=True):
        self.model = model
        self.use_cuda = use_cuda

    def reconstruction(self, data, size=(8, 8)):
        self.model.eval()
        input_data = torch.Tensor(data)

        if self.use_cuda:
            input_data = input_data.cuda()
        recon_data, *_ = self.model(input_data)
        self.model.train()

        # Upper half of plot will contain data, bottom half will contain
        # reconstructions
        num_images = size[0] * size[1] // 2
        originals = input_data[:num_images].cpu()
        reconstructions = recon_data.view(-1, *self.model.img_size)[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])

        return make_grid(comparison.data, nrow=size[0])


    def sample(self, n_sample):
        samples = np.random.normal(size=(n_sample, self.model.latent_dim))
        st = np.std(samples, axis=1)
        samples = torch.Tensor(samples)
        samples = samples.cuda()
        return make_grid(self.model.decode(samples).cpu().detach())