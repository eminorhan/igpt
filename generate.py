import argparse
import torch
import torchvision
from mingpt.utils import ImageDataset, generate_samples, generate_from_half, generate_chimera
from mingpt.model import GPT, GPTConfig 
from torch.utils.data.dataloader import DataLoader

parser = argparse.ArgumentParser(description='Generate samples from an Image GPT')
parser.add_argument('--data_cache', default='', type=str, help='Cache path for the stored training set')
parser.add_argument('--model_cache', default='', type=str, help='Cache path for the stored model')
parser.add_argument('--condition', default='chimera', type=str, help='Generation condition', choices=['uncond', 'half', 'chimera'])

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
model.load_state_dict(model_ckpt['model_state_dict'])

if torch.cuda.is_available():
    model = model.cuda()

if args.condition == 'uncond':
    # generate some samples unconditionally
    print("Generating unconditional samples")
    generate_samples(model, train_dataset, 32)
elif args.condition == 'half' or args.condition == 'chimera':
    # generate samples conditioned on upper half
    img_dir = '/scratch/eo41/minGPT/frames_for_half_1'
    print("Generating samples from upper half of images at {}".format(img_dir))
    x_data = torchvision.datasets.ImageFolder(img_dir, torchvision.transforms.Resize((train_dataset.d_img, train_dataset.d_img)))
    x_dataset = ImageDataset(x_data, train_dataset.d_img, train_dataset.clusters)
    x_loader = DataLoader(x_dataset, shuffle=True, pin_memory=True, batch_size=6, num_workers=8)  # TODO: better way to handle the parameters here

    for _, (x, _) in enumerate(x_loader):
        generate_from_half(x, model, train_dataset)