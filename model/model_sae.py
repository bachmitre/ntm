from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from collections import defaultdict

import numpy as np
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

from model_utils import SinkhornDistance, SimpleDataset
from model import TopicModel
import pandas as pd


class Encoder(nn.Module):
    """
    Topic model using Auto Encoder
    This is the encoder: a stack of dense layers going from input dimension (length of vocabulary / bow)
    to number of topics (=output dimension)
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation='relu', dropout=0.2):

        super(Encoder, self).__init__()

        self.activation = {'relu': nn.ReLU(), 'softplus': nn.Softplus()}[activation]

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), self.activation)
        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))])
        )

        # dropout
        self.output_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], output_size), self.activation)
        self.dropout = nn.Dropout(p=dropout)

        # batch normalization
        self.batchnorm = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hiddens(x)
        x = self.dropout(x)
        x = self.batchnorm(self.output_layer(x))
        return x


class Decoder(nn.Module):
    """
    Topic model using Auto Encoder
    This is the decoder part: encode the input, softmax it (resulting in the topic assignment), and reconstruct (=decode) it from that
    In order to enforce a certain (dirichlet or other) distribution of the topics the sinkhorn distance is used
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation='relu', dropout=0.2, dist_epsilon=0.1, dist_iterations=50, dist_reduction='mean', rand_mix=0.5):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.rand_mix = rand_mix

        self.activation = {'relu': nn.ReLU(), 'softplus': nn.Softplus()}[activation]

        # encoder
        self.encoder = Encoder(input_size, output_size, hidden_sizes, activation=activation, dropout=0.2)

        # output layer as trainable tensor
        self.output_layer = torch.Tensor(output_size, input_size)
        if torch.cuda.is_available():
            self.output_layer = self.output_layer.cuda()
        self.output_layer = nn.Parameter(self.output_layer)
        nn.init.xavier_uniform_(self.output_layer)

        # dropout
        self.dropout = nn.Dropout(p=dropout)

        # batch normalization
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # sinkhorn distance
        self.sinkhorn = SinkhornDistance(eps=dist_epsilon, max_iter=dist_iterations, reduction=dist_reduction)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.dropout(x)
        topics = self.encoder(x)
        topics = F.softmax(topics, dim=1)
        topics = self.dropout(topics)
        # during training, add a protion of a random (dirichlet) distributed vector to the topic distribution
        # this further enforces the desired topic distribution to adapt to the prior
        if self.training:
            topics = (1 - self.rand_mix) * topics + self.rand_mix * self.sample('dirichlet', batch_size).cuda()
        x = F.softmax(self.beta_batchnorm(torch.matmul(topics, self.output_layer)), dim=1)
        return x, topics

    def sample(self, dist='dirichlet', batch_size=256, dirichlet_alpha=0.2, ori_data=None, z_dim=None):
        """
        sample from a given distribution
        """
        if not z_dim:
            z_dim = self.output_size

        if dist == 'dirichlet':
            z_true = np.random.dirichlet(
                np.ones(z_dim) * dirichlet_alpha, size=batch_size)
            z_true = torch.from_numpy(z_true).float()
            return z_true

        elif dist == 'gaussian':
            z_true = np.random.randn(batch_size, z_dim)
            z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
            return z_true

        elif dist == 'gmm_std':
            odes = np.eye(z_dim) * 20
            ides = np.random.randint(low=0, high=z_dim, size=batch_size)
            mus = odes[ides]
            sigmas = np.ones((batch_size, z_dim)) * 0.2 * 20
            z_true = np.random.normal(mus, sigmas)
            z_true = F.softmax(torch.from_numpy(z_true).float(), dim=1)
            return z_true

        elif dist == 'gmm_ctm' and ori_data != None:
            with torch.no_grad():
                hid_vecs = self.inf_net(ori_data).cpu().numpy()
                gmm = GaussianMixture(n_components=z_dim, covariance_type='full', max_iter=200)
                gmm.fit(hid_vecs)
                gmm_spls, _spl_lbls = gmm.sample(n_samples=len(ori_data))
                theta_prior = torch.from_numpy(gmm_spls).float()
                theta_prior = F.softmax(theta_prior, dim=1)
                return theta_prior

        else:
            return self.sample(dist='dirichlet', batch_size=batch_size)


class TopicModelSae(TopicModel):
    """
    Auto Encoder Topic Model using sinkhorn distance (in addition to reconstruction error) as loss to enforce desired
    topic distribution
    """
    def __init__(
            self,
            vocab=None,
            n_components=10,
            hidden_sizes=(100, 100),
            activation='softplus',
            dropout=0.2,
            batch_size=64,
            lr=2e-3,
            momentum=0.99,
            n_epochs=10,
            dist='dirichlet',
            dist_epsilon=0.1,
            dist_iterations=50,
            dist_reduction='mean',
            rand_mix=0.5
    ):
        self.vocab = vocab
        self.input_size = len(vocab)
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.num_epochs = n_epochs
        self.dist = dist

        # encoder / decoder model
        self.model = Decoder(
            self.input_size,
            n_components,
            hidden_sizes,
            activation,
            dropout,
            dist_epsilon,
            dist_iterations,
            dist_reduction,
            rand_mix
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))

        self.best_loss_train = float('inf')
        self.best_components = None

        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        if self.USE_CUDA:
            self.model = self.model.cuda()

    def _loss(self, inputs, word_dists, prior_mean, posterior_mean):
        # sinkhorn distribution distance between prior and posterior
        KL, _, _ = self.model.sinkhorn(prior_mean, posterior_mean)
        # reconstruction error
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        loss = KL + RL
        return loss.sum()

    def _train_epoch(self, loader):
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in loader:
            X = batch_samples['X']

            if self.USE_CUDA:
                X = X.cuda()

            self.model.zero_grad()

            # reconstructed X and topic distribution
            X_rec, topics_dist = self.model(X)

            # sample from the desired distribution
            sample_dist = self.model.sample(dist=self.dist, batch_size=len(X), ori_data=X)
            if self.USE_CUDA:
                sample_dist = sample_dist.cuda()

            # backward pass
            loss = self._loss(
                X,
                X_rec,
                sample_dist,
                topics_dist
            )
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss

    def fit(self, train_data, save_dir=None, verbose=False):

        self.train_data = SimpleDataset(train_data)

        if verbose:
            print(self.model)
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total Params: {}".format(total_params))

        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )

        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            # report
            if verbose:
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(train_data) * self.num_epochs, train_loss, e - s))

            # save best
            if train_loss < self.best_loss_train:
                self.best_loss_train = train_loss
                self.best_components = self.model.output_layer

    def get_topics(self, n_terms=10):
        component_dists = self.best_components
        topics = defaultdict(list)
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], n_terms)
            component_words = [self.vocab[idx] for idx in idxs.cpu().numpy()]
            topics[i] = component_words
        return pd.DataFrame(topics).transpose()

    def transform(self, dataset):

        self.model.eval()

        loader = DataLoader(
            SimpleDataset(dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0)

        preds = []

        with torch.no_grad():
            for batch_samples in loader:

                X = batch_samples['X']

                if self.USE_CUDA:
                    X = X.cuda()

                # forward pass
                self.model.zero_grad()
                _, topics = self.model(X)

                preds += [topics]
            preds = torch.cat(preds, dim=0)

        return preds
