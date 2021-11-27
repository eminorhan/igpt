import os
import builtins
import argparse
import torch
import torchvision
import torch.distributed as dist
from mingpt.utils import ImageDataset, make_dictionary
from mingpt.model import GPT, GPTConfig 
from mingpt.trainer import Trainer, TrainerConfig

parser = argparse.ArgumentParser(description='Train an Image GPT')
parser.add_argument('data', metavar='DIR', help='path to frames')
parser.add_argument('--save_dir', default='', type=str, help='model save directory')
parser.add_argument('--d_img', default=64, type=int, help='image size (pixels)')
parser.add_argument('--dict_size', default=512, type=int, help='dictionary size')
parser.add_argument('--n_layer', default=24, type=int, help='number of layers')
parser.add_argument('--n_head', default=8, type=int, help='number of attention heads')
parser.add_argument('--n_embd', default=512, type=int, help='embedding dimensionality')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size per gpu')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--subject', default='SAY', choices=['ImageNet', 'SAY', 'S', 'A', 'Y', 'brady'], help='subject')
parser.add_argument('--data_cache', default='', type=str, help='Cache path for the training set for quicker initialization')
parser.add_argument('--resume', default='', type=str, help='Model path for resuming training')
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

args = parser.parse_args()
print(args)

# DDP setting
if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
args.distributed = args.world_size > 1

if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
        args.rank = args.local_rank
        args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

print('Running on {} GPUs total'.format(args.world_size))
model_name = 'model_{}l_{}h_{}e_{}b_{}d_{}lr_{}ep_{}.pt'.format(
    args.n_layer, args.n_head, args.n_embd, args.world_size * args.batch_size, args.d_img, args.lr, args.epochs, args.subject
    )
ckpt_path = os.path.join(args.save_dir, model_name)
print('The model will be saved to', ckpt_path)

if args.data_cache and os.path.exists(args.data_cache):
    print("Loading training dataset from {}".format(args.data_cache))
    train_dataset = torch.load(args.data_cache)
else:
    print("Building training dataset from scratch")
    # adjust transforms as needed
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.d_img)
        ])
    train_data = torchvision.datasets.ImageFolder(args.data, train_transforms)
    cluster_centers = make_dictionary(train_data, args.dict_size, args.d_img)
    train_dataset = ImageDataset(train_data, args.d_img, cluster_centers)
    torch.save(train_dataset, args.data_cache)

# some sanity checks
print('Training data size:', len(train_dataset))
print('Dictionary shape:', train_dataset.clusters.shape)
print('Example flattened image shape:', train_dataset[0][0].shape)
print('Example flattened image:', train_dataset[0][0])  # one example image flattened out into integers

# set up model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
model = GPT(mconf)

if args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
else:
    model = torch.nn.DataParallel(model.cuda())

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=0.0)

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
tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, ckpt_path=ckpt_path, num_workers=4)
trainer = Trainer(model, optimizer, train_dataset, tconf)
trainer.train(args)