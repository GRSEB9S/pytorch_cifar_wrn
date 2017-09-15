import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from wrn_selector import WideResNetSelector

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNetSelector', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.name = str(args.layers)+'-'+str(args.widen_factor)+'-'+str(args.droprate)+'-'+'selector-'+args.name
    #direcotry example: runs_cifar10/0/28-4-WideResNet...model
    if args.tensorboard: configure("runs_"+args.dataset+"/selector")

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(
        						Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
        						(4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    """
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # create model
    model = WideResNetSelector(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        """
        val_loader = torch.utils.data.DataLoader(
            CIFAR10(args.target_label, '.', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        """
        adjust_learning_rate(optimizer, epoch+1)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        # modifying...prec1 = validate(val_loader, model, criterion, epoch)
        #average_pos, average_neg = validate_single(val_loader, model, criterion, epoch)
        prec1 = validate(val_loader, model, criterion, epoch)
        # remember best prec@1 and save checkpoint
        #average_error = 2.0-average_pos + 2.0-average_neg#Max=2.0 due to tanh
        is_best =  prec1 < best_prec1
        best_prec1 = max(best_prec1, prec1)
        #best_error = min(best_error, average_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('***Best Prec: ' +  str(best_prec1) + "***")

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def train_single(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    average_pos = AverageMeter()
    average_neg = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        #print(input.size())
        #print(target.size())
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)#TO BE CONTINUED
        
        # measure accuracy and record loss
        #prec1 = accuracy(output.data, target, topk=(1,))[0]
        pos_activation, pos_n = activation_level(output.data, target)
        average_pos.update(pos_activation, pos_n)
        neg_activation, neg_n = activation_level(output.data, target, positive=False)
        average_neg.update(neg_activation, neg_n)

        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Pos_ave ({average_pos.avg:.3f})\t'
                  'Neg_ave ({average_neg.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, average_pos=average_pos, average_neg=average_neg))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', epoch, losses.avg)
        log_value('train_activation', epoch, average_pos.avg, average_neg.avg)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg

def validate_single(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    average_pos = AverageMeter()
    average_neg = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure positive/negative activation and record loss
        #prec1 = accuracy(output.data, target, topk=(1,))[0]
        pos_activation, pos_n = activation_level(output.data, target)
        average_pos.update(pos_activation, pos_n)
        neg_activation, neg_n = activation_level(output.data, target, positive=False)
        average_neg.update(neg_activation, neg_n)
        

        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Pos_ave ({average_pos.avg:.3f})\t'
                  'Neg_ave ({average_neg.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      average_pos=average_pos, average_neg=average_neg))

    #print(' '.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', epoch, losses.avg)
        log_value('val_activation', epoch, average_pos.avg, average_neg.avg)
    return average_pos.avg, average_neg.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = os.path.join("runs_"+args.dataset, "selector")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory ,filename)
    torch.save(state, filename)
    if is_best:
        print("Saving best model...")
        shutil.copyfile(filename, os.path.join(directory , args.name+'_model_best.pth.tar'))
        print("Best model saved success")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = args.lr * ((0.5 ** int(epoch >= 5)) * (0.5 ** int(epoch >= 10))* (0.5 ** int(epoch >= 15))*((0.5 ** int(epoch >= 20))))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def activation_level(output, target, positive=True):
    """Get the average activation for selected label"""
    target_label = 1 if positive else 0;
    #output = output.view(-1)
    #print(target.size())
    #print(output.size())
    output_normal = output[:,target_label]
    output_opposite = output[:,1-target_label]
    #print(output.size())
    selected_outputs_normal = output_normal[target == target_label]
    selected_outputs_opposite = output_opposite[target == target_label]
    if(len(selected_outputs_normal.shape) == 0):
        return 0,0
    norm_mean = torch.mean(selected_outputs_normal)
    oppo_mean = torch.mean(selected_outputs_opposite)
    if(norm_mean < -1 or norm_mean > 1 or oppo_mean < -1 or oppo_mean > 1):
        print("norm_mean:" + str(norm_mean) + " oppo_mean:"+str(oppo_mean))
    #print(selected_outputs_normal.shape)
    #print(selected_outputs_normal.size(0))
    return norm_mean-oppo_mean, selected_outputs_normal.shape[0]

if __name__ == '__main__':
    main()
