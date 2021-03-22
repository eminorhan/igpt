import os
import argparse
import logging
import torch
import torchvision
import numpy as np
from mingpt.utils import ImageDataset, make_dictionary, set_seed, generate_samples
from mingpt.model import GPT, GPTConfig 
from mingpt.trainer import Trainer, TrainerConfig
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Train an Image GPT on SAYCam')
parser.add_argument('data', metavar='DIR', help='path to SAYCam frames')
parser.add_argument('--save_dir', default='', type=str, help='model save directory')
parser.add_argument('--d_img', default=36, type=int, help='image size (pixels)')
parser.add_argument('--dict_size', default=384, type=int, help='dictionary size')
parser.add_argument('--n_layer', default=12, type=int, help='number of layers')
parser.add_argument('--n_head', default=8, type=int, help='number of attention heads')
parser.add_argument('--n_embd', default=512, type=int, help='embedding dimensionality')
parser.add_argument('--epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--subject', default='A', choices=['SAY', 'S', 'A', 'Y'], help='subject')
parser.add_argument('--data_cache', default='', type=str, help='Cache path for the training set for quicker initialization')

args = parser.parse_args()
print(args)

set_seed(42)

ckpt_path = os.path.join(args.save_dir, 'model_{}.pt'.format(args.subject))
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

if args.data_cache and os.path.exists(args.data_cache):
    print("Loading training dataset from {}".format(args.data_cache))
    train_dataset = torch.load(args.data_cache)
else:
    print("Building training dataset from scratch")
    train_data = torchvision.datasets.ImageFolder(args.data, transforms.Resize(args.d_img))
    cluster_centers = make_dictionary(train_data, args.dict_size, args.d_img)
    train_dataset = ImageDataset(train_data, args.d_img, cluster_centers)
    torch.save(train_dataset, args.data_cache)

# some sanity checks
print('Training data size:', len(train_dataset))
print('Dictionary shape:', train_dataset.clusters.shape)
print('Example flattened image shape:', train_dataset[0][0].shape)
print('Example flattened image:', train_dataset[0][0])  # one example image flattened out into integers

## set up model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, 
                n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
model = GPT(mconf)

tokens_per_epoch = len(train_dataset) * train_dataset.block_size

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=3e-3,
                    betas = (0.9, 0.95), weight_decay=0,
                    lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=args.epochs*tokens_per_epoch,
                    ckpt_path=ckpt_path,
                    num_workers=8)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()

# load the state of the best model we've seen based on early stopping
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint)

# generate some samples
generate_samples(model, train_dataset, 32)