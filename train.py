import os
import argparse
import logging
import torch
import torchvision
from mingpt.utils import ImageDataset, make_dictionary, set_seed, generate_samples
from mingpt.model import GPT, GPTConfig 
from mingpt.trainer import Trainer, TrainerConfig

parser = argparse.ArgumentParser(description='Train an Image GPT on SAYCam')
parser.add_argument('data', metavar='DIR', help='path to SAYCam frames')
parser.add_argument('--save_dir', default='', type=str, help='model save directory')
parser.add_argument('--d_img', default=48, type=int, help='image size (pixels)')
parser.add_argument('--dict_size', default=512, type=int, help='dictionary size')
parser.add_argument('--n_layer', default=24, type=int, help='number of layers')
parser.add_argument('--n_head', default=8, type=int, help='number of attention heads')
parser.add_argument('--n_embd', default=512, type=int, help='embedding dimensionality')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--subject', default='A', choices=['SAY', 'S', 'A', 'Y'], help='subject')
parser.add_argument('--data_cache', default='', type=str, help='Cache path for the training set for quicker initialization')
parser.add_argument('--resume', default='', type=str, help='Model path for resuming training')

args = parser.parse_args()
print(args)

set_seed(42)

ckpt_path = os.path.join(args.save_dir, 'model_24l_8h_512e_32b_{}.pt'.format(args.subject))
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

if args.data_cache and os.path.exists(args.data_cache):
    print("Loading training dataset from {}".format(args.data_cache))
    train_dataset = torch.load(args.data_cache)
else:
    print("Building training dataset from scratch")
    train_data = torchvision.datasets.ImageFolder(args.data, torchvision.transforms.Resize(args.d_img))
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
model = torch.nn.DataParallel(model).cuda()
optimizer = torch.optim.Adam(model.module.parameters(), 0.0005, weight_decay=0.0)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("=> no checkpoint found at '{}', will train from scratch".format(args.resume))

tokens_per_epoch = len(train_dataset) * train_dataset.block_size

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, ckpt_path=ckpt_path, num_workers=8)
trainer = Trainer(model, optimizer, train_dataset, None, tconf)
trainer.train()

# load the state of the best model we've seen based on early stopping
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint)

# generate some samples
generate_samples(model, train_dataset, 32)