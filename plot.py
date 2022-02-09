import os
import torch

path = '/scratch/eo41/minGPT/data_model_cache/brady_1_0_imagenet'
files = os.listdir(path)
files.sort()

for file in files:
    file_path = os.path.join(path, file)
    checkpoint = torch.load(file_path)
    train_loss = checkpoint['train_loss']
    print('File:', file)
    print('Train loss is ', train_loss)