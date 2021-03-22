import argparse
import torch
import torchvision
from mingpt.utils import ImageDataset, generate_from_half
from mingpt.model import GPT, GPTConfig 
from torch.utils.data.dataloader import DataLoader

parser = argparse.ArgumentParser(description='Generate samples from an Image GPT')
parser.add_argument('data', metavar='DIR', help='path to half frames')
parser.add_argument('--data_cache', default='', type=str, help='Cache path for the stored training set')
parser.add_argument('--model_cache', default='', type=str, help='Cache path for the stored model')

args = parser.parse_args()
print(args)

# load the data
train_dataset = torch.load(args.data_cache)

## set up model (TODO: better way to handle the model config)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_layer=12, n_head=8, n_embd=512)
model = GPT(mconf)

# load the model
print("Loading model")
model_ckpt = torch.load(args.model_cache)
model.load_state_dict(model_ckpt)

if torch.cuda.is_available():
    model = model.cuda()

# # generate some samples
# print("Generating samples")
# generate_samples(model, train_dataset, train_dataset.clusters, train_dataset.vocab_size, 32)

# generate samples from half
print("Generating samples from half")
x_data = torchvision.datasets.ImageFolder(args.data, torchvision.transforms.Resize(train_dataset.d_img))
x_dataset = ImageDataset(x_data, train_dataset.d_img, train_dataset.clusters)
x_loader = DataLoader(x_dataset, shuffle=True, pin_memory=True, batch_size=5, num_workers=8)

for _, (x, _) in enumerate(x_loader):
    print(x)
    generate_from_half(x, model, train_dataset, train_dataset.clusters, train_dataset.vocab_size)