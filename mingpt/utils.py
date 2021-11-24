import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    """
    wrap up the dataset into our own, which will convert images into sequences of integers
    """
    def __init__(self, pt_dataset, d_img, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.d_img = d_img
        self.perm = torch.arange(self.d_img * self.d_img) if perm is None else perm
        
        self.vocab_size = clusters.size(0)
        self.block_size = self.d_img * self.d_img - 1
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1, 3) # flatten out all pixels
        x = x[self.perm].float() # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1) # cluster assignments
        return a[:-1], a[1:] # always just predict the next one in the sequence

class ImageDatasetWithLabels(Dataset):
    """
    wrap up the dataset into our own, which will convert images into sequences of integers
    """
    def __init__(self, pt_dataset, d_img, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.d_img = d_img
        self.perm = torch.arange(self.d_img * self.d_img) if perm is None else perm
        
        self.vocab_size = clusters.size(0)
        self.block_size = self.d_img * self.d_img - 1
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1, 3)  # flatten out all pixels
        x = x[self.perm].float()  # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1) # cluster assignments
        return a[:-1], y  # always just predict the next one in the sequence

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    return x

def make_dictionary(train_data, dict_size, d_img):
    # get random 2 pixels per image and stack them all up as rgb values to get half a million random pixels
    pluck_rgb = lambda x: np.array(x).reshape(d_img**2, 3)[np.random.permutation(d_img**2)[:2], :]
    px = np.concatenate([pluck_rgb(x) for x, y in train_data], axis=0)
    px = np.float32(px)
    print('Building the dictionary')
    print('Pixel stack shape:', px.shape)

    ## compute dictionary
    kmeans = MiniBatchKMeans(n_clusters=dict_size, random_state=0, batch_size=128) 
    kmeans.fit(px)

    cluster_centers = kmeans.cluster_centers_
    print('Cluster centers shape:', cluster_centers.shape)

    cluster_centers = torch.from_numpy(cluster_centers)

    return cluster_centers
    
def generate_samples(model, train_dataset, n_samples):
    # to sample we also have to technically "train" a separate model for the first token in the sequence
    # we are going to do so below simply by calculating and normalizing the histogram of the first token
    cluster_centers = train_dataset.clusters
    dict_size = train_dataset.vocab_size

    counts = torch.ones(dict_size)  # start counts as 1 not zero, this is called "smoothing"
    rp = torch.randperm(len(train_dataset))
    nest = min(1000, len(train_dataset))  # how many images to use for the estimation
    for i in range(nest):
        a, _ = train_dataset[int(rp[i])]
        t = a[0].item()  # index of first token in the sequence
        counts[t] += 1
    prob = counts / counts.sum()

    ## sample some generated images
    start_pixel = np.random.choice(np.arange(cluster_centers.size(0)), size=(n_samples, 1), replace=True, p=prob.numpy())
    start_pixel = torch.from_numpy(start_pixel)
    if torch.cuda.is_available():
        start_pixel = start_pixel.cuda()

    print('Starting sampling.')    
    pixels = sample(model, start_pixel, train_dataset.d_img * train_dataset.d_img - 1, temperature=1.0, sample=True, top_k=100)

    # for visualization we have to invert the permutation used to produce the pixels
    iperm = torch.argsort(train_dataset.perm)

    ncol = 2 * int(np.sqrt(n_samples // 2))  # ncol:nrow = 2:1
    nrow = n_samples // ncol
    plt.figure(figsize=(2*ncol, 2*nrow))
    for i in range(n_samples):
        pxi = pixels[i][iperm] # note: undo the encoding permutation
        
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(cluster_centers[pxi].view(train_dataset.d_img, train_dataset.d_img, 3).numpy().astype(np.uint8))
        plt.axis('off')

    plt.savefig('samples.pdf', bbox_inches='tight')

    # visualize some of the learned positional embeddings, maybe they contain structure
    n_posembs = 8 * 8
    ncol_posembs = int(np.sqrt(n_posembs))
    nrow_posembs = n_posembs // ncol_posembs
    plt.figure(figsize=(ncol_posembs, nrow_posembs))
    for i in range(n_posembs):
        ci = model.pos_emb.data[0, :, i].cpu()
        zci = torch.cat((torch.tensor([0.0]), ci))  # pre-cat a zero
        rzci = zci[iperm]  # undo the permutation to recover the pixel space of the image
        
        plt.subplot(nrow_posembs, ncol_posembs, i+1)
        plt.imshow(rzci.view(train_dataset.d_img, train_dataset.d_img).numpy())
        plt.axis('off')

    plt.savefig('posembs.pdf', bbox_inches='tight')

def generate_from_half(x, model, train_dataset):
    """
    Generate the lower half given the upper half.
    """
    cluster_centers = train_dataset.clusters
    # TODO: better handle size here
    pixels_0 = sample(model, x[:, :2300].cuda(), 4, temperature=0.95, sample=True, top_k=100)
    pixels_1 = sample(model, x[:, :1152].cuda(), 1152, temperature=0.95, sample=True, top_k=100)
    pixels_2 = sample(model, x[:, :1152].cuda(), 1152, temperature=0.95, sample=True, top_k=100)
    pixels_3 = sample(model, x[:, :1152].cuda(), 1152, temperature=0.95, sample=True, top_k=100)
    pixels_4 = sample(model, x[:, :1152].cuda(), 1152, temperature=0.95, sample=True, top_k=100)

    # for visualization we have to invert the permutation used to produce the pixels
    iperm = torch.argsort(train_dataset.perm)

    ncol = 6
    nrow = 30 // ncol
    plt.figure(figsize=(16, 16))
    for i in range(6):
        pxi = pixels_0[i][iperm]  # note: undo the encoding permutation
        gen_img = cluster_centers[pxi].view(train_dataset.d_img, train_dataset.d_img, 3).cpu().numpy().astype(np.uint8)
        gen_img[23, :, :] = 0
        
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(gen_img)
        plt.axis('off')

    for i in range(6):
        pxi = pixels_1[i][iperm] # note: undo the encoding permutation
        gen_img = cluster_centers[pxi].view(train_dataset.d_img, train_dataset.d_img, 3).numpy().astype(np.uint8)
        gen_img[23, :, :] = 0

        plt.subplot(nrow, ncol, i+1+6)
        plt.imshow(gen_img)
        plt.axis('off')

    for i in range(6):
        pxi = pixels_2[i][iperm] # note: undo the encoding permutation
        gen_img = cluster_centers[pxi].view(train_dataset.d_img, train_dataset.d_img, 3).numpy().astype(np.uint8)
        gen_img[23, :, :] = 0

        plt.subplot(nrow, ncol, i+1+12)
        plt.imshow(gen_img)
        plt.axis('off')

    for i in range(6):
        pxi = pixels_3[i][iperm] # note: undo the encoding permutation
        gen_img = cluster_centers[pxi].view(train_dataset.d_img, train_dataset.d_img, 3).numpy().astype(np.uint8)
        gen_img[23, :, :] = 0
        
        plt.subplot(nrow, ncol, i+1+18)
        plt.imshow(gen_img)
        plt.axis('off')

    for i in range(6):
        pxi = pixels_4[i][iperm] # note: undo the encoding permutation
        gen_img = cluster_centers[pxi].view(train_dataset.d_img, train_dataset.d_img, 3).numpy().astype(np.uint8)
        gen_img[23, :, :] = 0
        
        plt.subplot(nrow, ncol, i+1+24)
        plt.imshow(gen_img)
        plt.axis('off')

    plt.savefig('samples_from_half.pdf', bbox_inches='tight')

def generate_chimera(x, model, train_dataset):
    """
    Generate chimera
    """
    cluster_centers = train_dataset.clusters

    pixels_0 = sample(model, x[:, :1291].cuda(), 5, temperature=1.0, sample=True, top_k=100)
    pixels_1 = sample(model, x[:, :648].cuda(), 648, temperature=1.0, sample=True, top_k=100)

    print(pixels_0.shape, pixels_1.shape)

    imgs, losses = [], []

    for i in range(5):
        for j in range(5):
            print(i, j)
            x = torch.unsqueeze(torch.cat((pixels_0[i, :648], pixels_1[j, 648:])), 0)
            with torch.no_grad():
                logits, loss = model(x[:, :-1], x[:, 1:])
                loss = loss.item()

            imgs.append(torch.squeeze(x, 0))
            losses.append(loss)

    # for visualization we have to invert the permutation used to produce the pixels
    iperm = torch.argsort(train_dataset.perm)

    ncol = 5
    nrow = 25 // ncol
    plt.figure(figsize=(16, 16))
    for i in range(25):
        pxi = imgs[i][iperm]  # note: undo the encoding permutation
        gen_img = cluster_centers[pxi].view(train_dataset.d_img, train_dataset.d_img, 3).numpy().astype(np.uint8)
        gen_img[18, :, :] = 0
        
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(gen_img)
        plt.title(str(losses[i])[:6])
        plt.axis('off')

    plt.savefig('samples_chimera.pdf', bbox_inches='tight')