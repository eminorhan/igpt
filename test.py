import os
import argparse
import torch
import numpy as np
from mingpt.utils import ImageDataset
from mingpt.model import GPT 
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(description='Test an Image GPT')
parser.add_argument('data', metavar='DIR', help='path to frames')
parser.add_argument('--data_cache', default='', type=str, help='Cache path for the stored test set')
parser.add_argument('--model_cache', default='', type=str, help='Cache path for the stored model')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--d_img', default=64, type=int, help='image size')
parser.add_argument('--save_name', default='', type=str, help='informative string for saving')

args = parser.parse_args()
print(args)

print("Loading model")
model_ckpt = torch.load(args.model_cache)
mconf = model_ckpt['model_config']
model = GPT(mconf)

# load the saved model params
model.load_state_dict(model_ckpt['model_state_dict'])

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

if args.data_cache and os.path.exists(args.data_cache):
    print("Loading training dataset from {}".format(args.data_cache))
    dataset = torch.load(args.data_cache)
else:
    print("Building training dataset from scratch")
    # adjust transforms as needed
    transforms = Compose([Resize(args.d_img)])
    data = ImageFolder(args.data, transforms)
    dataset = ImageDataset(data, args.d_img, model_ckpt['clusters'])
    torch.save(dataset, args.data_cache)

loader = DataLoader(dataset, shuffle=False, pin_memory=True, sampler=None, batch_size=args.batch_size, num_workers=args.num_workers)

unreduced_losses = []

with torch.no_grad():
    for it, (x, y) in enumerate(loader):
        # place data on the correct device
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # forward the model
        _, _, unreduced_loss = model(x, y)  # the first output returns the logits, the second mean loss over batch
        unreduced_losses.append(unreduced_loss.cpu().numpy())
    
    unreduced_losses = np.concatenate(unreduced_losses)
    np.save('losses_{}.npy'.format(args.save_name), unreduced_losses)