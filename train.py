import os
import builtins
import argparse
import torch
import torchvision
from mingpt.utils import ImageDataset, make_dictionary, generate_samples
from mingpt.model import GPT, GPTConfig 
from mingpt.trainer import Trainer, TrainerConfig
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Train an Image GPT')
parser.add_argument('data', metavar='DIR', help='path to frames')
parser.add_argument('--save_dir', default='', type=str, help='model save directory')
parser.add_argument('--d_img', default=48, type=int, help='image size (pixels)')
parser.add_argument('--dict_size', default=512, type=int, help='dictionary size')
parser.add_argument('--n_layer', default=24, type=int, help='number of layers')
parser.add_argument('--n_head', default=8, type=int, help='number of attention heads')
parser.add_argument('--n_embd', default=512, type=int, help='embedding dimensionality')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--subject', default='SAY', choices=['ImageNet', 'SAY', 'S', 'A', 'Y'], help='subject')
parser.add_argument('--data_cache', default='', type=str, help='Cache path for the training set for quicker initialization')
parser.add_argument('--resume', default='', type=str, help='Model path for resuming training')
parser.add_argument('--finetune', default=False, action='store_true', help='freeze trunk?')
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
ngpus_per_node = torch.cuda.device_count()

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

model_name = 'model_{}l_{}h_{}e_{}b_{}.pt'.format(args.n_layer, args.n_head, args.n_embd, args.batch_size, args.subject)
ckpt_path = os.path.join(args.save_dir, model_name)
print('The model will be saved to', ckpt_path)

if args.data_cache and os.path.exists(args.data_cache):
    print("Loading training dataset from {}".format(args.data_cache))
    train_dataset = torch.load(args.data_cache)
else:
    print("Building training dataset from scratch")
    train_data = torchvision.datasets.ImageFolder(args.data, torchvision.transforms.Resize((args.d_img, args.d_img)))
    if args.finetune:
        pretrain_dataset = torch.load('/scratch/eo41/minGPT/data_model_cache/data_SAY_half_fps.pth')  # TODO: handle this better (with a separate arg)
        cluster_centers = pretrain_dataset.clusters
    else:
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
    # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
else:
    raise NotImplementedError("Only DistributedDataParallel is supported.")

optimizer = torch.optim.Adam(model.parameters(), 0.0005, weight_decay=0.0)

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
trainer = Trainer(model, optimizer, train_dataset, None, tconf)
trainer.train(args)

# # TODO: we get a load error here.
# # load the state of the best model we've seen based on early stopping
# checkpoint = torch.load(ckpt_path)
# model.load_state_dict(checkpoint)

# # generate some samples
# generate_samples(model, train_dataset, 32)