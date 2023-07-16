import torch.nn as nn


NUM_CLASSES = 10


class Encoder(nn.Module):
    def __init__(self, c_in, c_hid, latent_dim):
        """
        Inputs:
            - c_in : Number of input channels of the image.
            - c_hid : Number of filters after the first convolution
            - latent_dim : Dimensionality of latent representation z
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid,
                      kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, c_in, c_hid, latent_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid), nn.ReLU()
            )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3,
                output_padding=1, padding=1, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                c_hid, c_in, kernel_size=3,
                output_padding=1, padding=1, stride=2
            ),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, c_hid, latent_dim, c_in=3):
        super().__init__()

        self.encoder = Encoder(c_in, c_hid, latent_dim)
        self.decoder = Decoder(c_in, c_hid, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class SimpleClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.clf = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(128),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        y_pred = self.clf(x)
        return y_pred
