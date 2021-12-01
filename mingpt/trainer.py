"""
Trainer code, simplified from Andrej Karpathy's minGPT
"""
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64

    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, optimizer, train_dataset, config):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.config = config
        self.train_loss = None

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = self.optimizer

        # save everything we need
        print('Saving to:', self.config.ckpt_path)
        torch.save({'model_state_dict': raw_model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'clusters': self.train_dataset.clusters,
                    'model_config': raw_model.model_config,
                    }, self.config.ckpt_path)

    def train(self, args):
        model, optimizer, config = self.model, self.optimizer, self.config
        model.train()

        sampler = DistributedSampler(self.train_dataset) if args.distributed else None
        loader = DataLoader(self.train_dataset, shuffle=(not args.distributed), pin_memory=True, 
                            sampler=sampler, batch_size=config.batch_size, num_workers=config.num_workers)

        for epoch in range(config.max_epochs):
            if args.distributed: 
                loader.sampler.set_epoch(epoch)

            losses = []

            for _, (x, y) in enumerate(loader):
                # place data on the correct device
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                # forward the model
                _, loss, _ = model(x, y)  # the first output returns the logits, last one returns unreduced losses
                losses.append(loss.item())

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                optimizer.step()

            self.train_loss = float(np.mean(losses))
            print('Epoch:', epoch, '|', 'Training loss:', self.train_loss)

        # save trained model, clusters, and final train loss
        if args.distributed:
            if args.rank == 0:
                self.save_checkpoint()
        else:
            self.save_checkpoint()