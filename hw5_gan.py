from audioop import bias
from doctest import OutputChecker
import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw5_utils import BASE_URL, download, GANDataset


class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # TODO: implement layers here
        self.layer1 = nn.Conv2d(in_channels = 1, out_channels = 2, kernel_size = 3, stride = 1, padding = 1) 
        self.layer2 = F.relu
        self.layer3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer4 = nn.Conv2d(in_channels = 2, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
        self.layer5 = F.relu
        self.layer6 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer7 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = 0)
        self.layer8 = F.relu
        # x = self.layer8.shape()  
        # i real do not know how i can get this 200, but bug in terminal told me it is256*200 , so we can know we have 200 feature hahahah
        self.layer9 = nn.Linear(200,1)   ####what about the input features?
        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for i in self.children():  ####search each layer one by one
            if hasattr(i, 'weight'):
                if i.weight is not None:
                    nn.init.kaiming_uniform_(i.weight.data, nonlinearity = 'relu')  ###kaiming uniform is better when relu exsit
            if hasattr(i, 'bias'):
                if i.bias is not None:
                    nn.init.constant_(i.bias.data, 0)  ### set it to 0
    def forward(self, x):
        # TODO: complete forward function
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = torch.flatten(x, start_dim = 1) ### dim0 is each entry, dim1 is vector, so flatten starting from dim 1
        x = self.layer9(x)
        return x


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()

        # TODO: implement layers here
        self.layer1 = nn.Linear(zdim,1568)
        # self.layer2 = F.leaky_relu(negative_slope = 0.2) ####how can i embed it into super
        self.layer3 = nn.Upsample(scale_factor = 2)
        self.layer4 = nn.Conv2d(in_channels = 32, out_channels = 16,kernel_size = 3, stride = 1, padding = 1)
        # self.layer5 = F.leaky_relu(negative_slope = 0.2)
        self.layer6 = nn.Upsample(scale_factor = 2)
        self.layer7 = nn.Conv2d(in_channels = 16, out_channels = 8,kernel_size = 3, stride = 1, padding = 1)
        # self.layer8 = F.leaky_relu(negative_slope = 0.2)
        self.layer9 = nn.Conv2d(in_channels = 8, out_channels = 1,kernel_size = 3, stride = 1, padding = 1)
        self.layer10 = nn.Sigmoid()
        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for i in self.children():  ####search each layer one by one
            if hasattr(i, 'weight'):
                if i.weight is not None:
                    nn.init.kaiming_uniform_(i.weight.data, nonlinearity = 'relu')  ###kaiming uniform is better when relu exsit
            if hasattr(i, 'bias'):
                if i.bias is not None:
                    nn.init.constant_(i.bias.data, 0)  ### set it to 0

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        x = z
        x = self.layer1(x)
        x = F.leaky_relu(x,0.2)
        x = torch.reshape(x, (-1, 32, 7, 7))
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.leaky_relu(x,0.2)
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.leaky_relu(x,0.2)
        x = self.layer9(x)
        x = self.layer10(x)
        return x

class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        # TODO: implement discriminator's loss function
        G = self.gen
        D = self.disc
        real_label = D(batch_data)
        assume_label = torch.ones(real_label.size())
        generate_real_label = D(G(z))
        assume_generate_real_label = torch.zeros(generate_real_label.size())
        real = torch.cat((real_label,generate_real_label), dim = 0)
        assume = torch.cat((assume_label,assume_generate_real_label), dim = 0)
        loss_method = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones(real.size()[1]))    ####combine bce with sigmoid
        loss = loss_method(real,assume)
        return loss


    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.
        Compute -\sum_z\log{D(G(z))} instead of \sum_z\log{1-D(G(z))}
        
        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        # TODO: implement generator's loss function
        G = self.gen
        D = self.disc
        generate_real_label = D(G(z))
        assume_generate_real_label = torch.ones(generate_real_label.size())
        loss_method = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones(generate_real_label.size()[1]))    ####combine bce with sigmoid
        loss = loss_method(generate_real_label,assume_generate_real_label)
        return loss

    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()
