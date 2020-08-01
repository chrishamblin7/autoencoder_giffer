import torch
from torch import nn
from torch.autograd import Variable

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(True),
            nn.Linear(400, 40))
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(True),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid())

    def reparametrize(self, mu, logvar):
        var = logvar.exp()
        std = var.sqrt()
        eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        return eps.mul(std).add(mu)

    def forward(self, x):
        h = self.encoder(x)
        mu = h[:, :20]
        logvar = h[:, 20:]
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def generation_with_interpolation(self, x_one, x_two, alpha):
        hidden_one = self.encoder(x_one)
        hidden_two = self.encoder(x_two)
        mu_one = hidden_one[:, :20]
        logvar_one = hidden_one[:, 20:]
        mu_two = hidden_two[:, :20]
        logvar_two = hidden_two[:, 20:]
        mu = (1 - alpha) * mu_one + alpha * mu_two
        logvar = (1 - alpha) * logvar_one + alpha * logvar_two
        z = self.reparametrize(mu, logvar)
        generated_image = self.decoder(z)
        return generated_image