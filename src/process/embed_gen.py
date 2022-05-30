import os
import numpy as np
from tqdm import tqdm


class EmbedGen():
    def __init__(self, config, vae_model, data, verbose=True):
        self.config = config
        self.verbose = verbose
        if self.verbose:
            print("Calculating embeddings for train dataset... ")
        self.x_train, self.y_train = self.get_embeddings(vae_model,
                                                         data.train_dataset_lstm)
        if self.verbose:
            print("Calculating embeddings for test dataset... ")
        self.x_test, self.y_test = self.get_embeddings(vae_model,
                                                       data.test_dataset_lstm)

    def get_embeddings(self, vae_model, dataset):
        embeddings = np.zeros((dataset.shape[0],
                              self.config['N_WIN'],
                              self.config['LATENT_DIM']))
        if self.verbose:
            for i in tqdm(range(dataset.shape[0])):
                embeddings[i] = vae_model.encoder_predict(dataset[i])
        else:
            for i in range(dataset.shape[0]):
                embeddings[i] = vae_model.encoder_predict(dataset[i])
        x_embedding = embeddings[:, :(self.config['N_WIN'] - 1)]
        y_embedding = embeddings[:, 1:]

        return x_embedding, y_embedding

    def print_info(self):
        print(f"\
            Embeddings Shape\n\
            ================\n\
            x_train:                {self.x_train.shape}\n\
            y_train:                {self.y_train.shape}\n\
            x_test:                 {self.x_test.shape}\n\
            y_test:                 {self.y_test.shape}\n\
        ")


class EmbedGenTL():
    def __init__(self, config, vae_model, data, verbose=True):
        self.config = config
        self.verbose = verbose
        if self.verbose:
            print("Calculating embeddings for train dataset... ")
        self.x_train, self.y_train = self.get_embeddings(vae_model,
                                                         data.train_dataset_lstm)

    def get_embeddings(self, vae_model, dataset):
        embeddings = np.zeros((dataset.shape[0],
                              self.config['N_WIN'],
                              self.config['LATENT_DIM']))
        if self.verbose:
            for i in tqdm(range(dataset.shape[0])):
                embeddings[i] = vae_model.encoder_predict(dataset[i])
        else:
            for i in range(dataset.shape[0]):
                embeddings[i] = vae_model.encoder_predict(dataset[i])
        x_embedding = embeddings[:, :(self.config['N_WIN'] - 1)]
        y_embedding = embeddings[:, 1:]

        return x_embedding, y_embedding

    def print_info(self):
        print(f"\
            Embeddings Shape\n\
            ================\n\
            x_train:                {self.x_train.shape}\n\
            y_train:                {self.y_train.shape}\n\
        ")
