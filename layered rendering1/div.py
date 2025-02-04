import torch.nn as nn

import numpy as np
# %%
class Generator(nn.Module):
    '''
    pure Generator structure

    '''

    def __init__(self, image_size=64, z_dim=100, conv_dim=64, channels=1):

        super(Generator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.z_dim = z_dim

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8

        self.l1 = nn.Sequential(
            # input is Z, going into a convolution.
            nn.ConvTranspose2d(self.z_dim, conv_dim * \
                               mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(
                curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(
                curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(
                curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)  # (*, 100, 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        out = self.last(out)

        return out

# %%
class Discriminator(nn.Module):
    '''
    pure discriminator structure

    '''

    def __init__(self, image_size=64, conv_dim=64, channels=1):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.channels = channels

        self.l1 = nn.Sequential(
            nn.Conv2d(self.channels, conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = conv_dim

        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2

        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2

        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2

        # output layers
        self.last_adv = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 4, 1, 0, bias=False),
            # without sigmoid, used in the loss funciton
        )

    def forward(self, x):
        out = self.l1(x)  # (*, 64, 32, 32)
        out = self.l2(out)  # (*, 128, 16, 16)
        out = self.l3(out)  # (*, 256, 8, 8)
        out = self.l4(out)  # (*, 512, 4, 4)

        validity = self.last_adv(out)  # (*, 1, 1, 1)

        return validity.squeeze()


# %%
import sys
sys.path.append('..')
import pprint
from torchinfo import summary

if __name__ == "__main__":

    G = Generator().cuda()
    D = Discriminator().cuda()

    # get model summary as string
    G_model_stats = summary(G, (64, 100))
    D_model_stats = summary(D, (64, 1, 64, 64))

    pprint.pprint(G_model_stats)
    pprint.pprint(D_model_stats)

    summary_str = (G_model_stats), (D_model_stats)
    with open('model_structure.log', 'w') as tf:
        pprint.pprint(summary_str, stream=tf)

# %%