import time
import argparse
import torch
import torchvision
import numpy as np
from mingpt.utils import ImageDataset, ImageDatasetWithLabels, generate_samples
from mingpt.model import GPT, GPTConfig, LinearProbeGPT 
from torch.utils.data.dataloader import DataLoader

parser = argparse.ArgumentParser(description='Evaluate model on eval data')
parser.add_argument('data', metavar='DIR', help='path to labeled S frames')
parser.add_argument('--traindata_cache', default='', type=str, help='Cache path for the stored training set')
parser.add_argument('--model_cache', default='', type=str, help='Cache path for the stored model')
parser.add_argument('--num_classes', default=26, type=int, help='Number of classes in downstream classification task')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--probe_layer', default=5, type=int, help='probe layer', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def freeze_trunk(model):
    '''Helper function for setting body to non-trainable'''
    for param in list(model.parameters())[:-2]:
        param.requires_grad = False


def load_split_train_test(labeled_s_dataset, batch_size, subsample=False, workers=8, train_frac=0.5):

    num_train = len(labeled_s_dataset)

    print('Total data size is', num_train)

    indices = list(range(num_train))
    split = int(np.floor(train_frac * num_train))
    np.random.shuffle(indices)

    if subsample:
        num_data = int(0.1 * num_train)
        train_idx, test_idx = indices[:(num_data // 2)], indices[(num_data // 2):num_data]
    else:
        train_idx, test_idx = indices[:split], indices[split:]

    print('Training data size is', len(train_idx))
    print('Test data size is', len(test_idx))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(labeled_s_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(labeled_s_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=test_sampler)

    return trainloader, testloader

def train(train_loader, model, criterion, optimizer, epoch, print_freq=100):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return top1.avg.cpu().numpy()

def validate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)

            preds = np.argmax(output.cpu().numpy(), axis=1)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg, preds, target.cpu().numpy(), images.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    # load the training data
    print("Loading data")
    train_dataset = torch.load(args.traindata_cache)

    # build the labeled S dataset using the dictionary learned over training data
    labeled_s_data = torchvision.datasets.ImageFolder(args.data, torchvision.transforms.Resize(train_dataset.d_img))
    labeled_s_dataset = ImageDatasetWithLabels(labeled_s_data, train_dataset.d_img, train_dataset.clusters)

    ## set up model (TODO: better way to handle the model config)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_layer=12, n_head=8, n_embd=512)
    model = GPT(mconf)

    # load the model
    print("Loading model")
    checkpoint = torch.load(args.model_cache)
    model.load_state_dict(checkpoint['model_state_dict'])

    prly = args.probe_layer
    head = torch.nn.Linear(in_features=512, out_features=args.num_classes, bias=True)  # TODO: better way to handle the model config (512 should be the same as n_embd)
    model = LinearProbeGPT(model.tok_emb, model.pos_emb, model.drop, model.blocks[:prly], model.blocks[prly+1].ln1, head)

    print(model)
    
    freeze_trunk(model)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    train_loader, test_loader = load_split_train_test(labeled_s_dataset, args.batch_size)
    acc1_list = []
    val_acc1_list = []

    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 0.0005, weight_decay=0.0)

    for epoch in range(1, args.epochs+1):
        # train for one epoch
        acc1 = train(train_loader, model, criterion, optimizer, epoch)
        acc1_list.append(acc1)

    # validate at end of epoch
    val_acc1, preds, target, images = validate(test_loader, model)
    val_acc1_list.append(val_acc1)

    torch.save({'acc1_list': acc1_list, 'val_acc1_list': val_acc1_list}, 'testrun2.tar')