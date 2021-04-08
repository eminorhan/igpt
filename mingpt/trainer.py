"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64

    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, optimizer, train_dataset, test_dataset, config):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = self.optimizer
        logger.info("saving %s", self.config.ckpt_path)
        torch.save({'model_state_dict': raw_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, self.config.ckpt_path)

    def train(self):
        model, optimizer, config = self.model, self.optimizer, self.config
        raw_model = model.module if hasattr(self.model, "module") else model

        def run_epoch(split, epoch):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers)

            losses = []
            print_freq = len(loader) // 100 

            for it, (x, y) in enumerate(loader):
                # place data on the correct device
                x = x.cuda()
                y = y.cuda()

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                # report progress
                if it % print_freq == 0:
                    print('Epoch:', epoch, '|', 'Iteration:', it, 'of', len(loader), '|', 'Training loss:', float(np.mean(losses)))
                    losses = []

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        for epoch in range(config.max_epochs):

            run_epoch('train', epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                # best_loss = test_loss
                self.save_checkpoint()