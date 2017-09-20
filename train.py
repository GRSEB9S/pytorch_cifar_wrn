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
from torchnet.meter import ConfusionMeter

from wideresnet import WideResNet

from cifar_10_custom import CIFAR10

TEST_DIR = "test_results"
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
parser.add_argument('--selective-train', dest='selective_train', action='store_true',
                    help='(fine tune the networks)whether to use selective training based on confusion matrix (default: False)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--target-label', default=0, type=int,
                    help='The selected target class to train')
parser.set_defaults(augment=True)

best_error = 4.0#max error is 4, sum error of pos & neg
best_prec1 = 0.0# accuracy, used for fine-tuning

def main():
    global args, best_error, best_prec1
    args = parser.parse_args()
    args.name = str(args.layers)+'-'+str(args.widen_factor)+'-'+str(args.droprate)+'-'+args.name
    args.lr = 0.000005 if args.selective_train else args.lr
    #direcotry example: runs_cifar10/0/28-4-WideResNet...model
    if args.tensorboard: configure("runs_"+args.dataset+"/%s"%(str(args.target_label)))#%(args.name))
   
    fine_tune_dir = None 
    fine_tune_run_dir = None
    fine_tune_test_dir = None
    if args.selective_train:
        fine_tune_dir = "finetune"
        if not os.path.exists(fine_tune_dir):
            os.makedirs(fine_tune_dir)
        fine_tune_run_dir = os.path.join(fine_tune_dir, "runs")
        if not os.path.exists(fine_tune_run_dir):
            os.makedirs(fine_tuen_run_dir)
        fine_tune_run_dir = os.path.join(fine_tune_dir, "runs", str(args.target_label))
        if not os.path.exists(fine_tune_run_dir):
            os.makedirs(fine_tune_run_dir)
        fine_tune_test_dir = os.path.join(fine_tune_dir, "test_results")
        if not os.path.exists(fine_tune_test_dir):
            os.makedirs(fine_tune_test_dir)

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
    torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    """
    
    """  
    torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    """
    val_loader = torch.utils.data.DataLoader(
            CIFAR10(args.target_label, [], '../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    #This keeps the original labels from 0~9    
    val_loader_original = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # create model
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 2 or 2,
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
            best_error = checkpoint['best_error']
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

    old_best_prec1 = 1.0
    for epoch in range(args.start_epoch, args.epochs):
        #For fine-tuning & selective training
        c_matrix = None
        """
        if args.selective_train:
            print("Training with selected proportion based on confusion matrix")
            confusion_file_path = os.path.join(TEST_DIR, "confusion_matrix")
            if os.path.isfile(confusion_file_path):
                c_matrix = torch.load(confusion_file_path)
                old_best_prec1 = compute_c_matrix_prec(c_matrix)
                print("Current overall accuracy:{0:.3f}".format(old_best_prec1))
        """
        sorted_train_list = []
        if args.selective_train:
            # Go through the training data first to find the least correct samples
            print("Finding least scored training samples...")
            train_loader_original = torch.utils.data.DataLoader(
                CIFAR10(args.target_label, sorted_train_list, '../data', train=True, download=True, 
                transform=transform_train, augment=False),
                batch_size=args.batch_size, shuffle=False, **kwargs)
            
            sorted_train_list = validate_train(train_loader_original, model, criterion, epoch)
 
        train_loader = torch.utils.data.DataLoader(
            CIFAR10(args.target_label, sorted_train_list[:10000], '../data', train=True, download=True, 
            transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        adjust_learning_rate(optimizer, epoch+1)

        # train for one epoch
        train_single(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        # modifying...prec1 = validate(val_loader, model, criterion, epoch)
        average_pos, average_neg, confusion_matrix, score_batch_list, accuracy_list = validate_single(val_loader, val_loader_original, model, criterion, epoch, update_c_matrix = args.selective_train, fine_tune_test_dir=fine_tune_test_dir)
        # remember best prec@1 and save checkpoint
        average_error = 2.0-average_pos + 2.0-average_neg#Max=2.0 due to tanh
        average_prec1 = compute_c_matrix_prec(confusion_matrix.value())
        is_best = average_error < best_error #average_error < best_error if not args.selective_train else average_prec1 > old_best_prec1
        best_error = min(best_error, average_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_error': best_error,
        }, is_best, fine_tune_run_dir)
        # Save score batch list if it's best
        if(args.selective_train):
            print("Confusion Matrix---")
            print(confusion_matrix.value())
            print("After, overall accuracy:{0:.3f}/{1:.3f}".format(average_prec1, old_best_prec1))
            print("Class Accuracies(Average)---")
            print([class_accuracy.avg for class_accuracy in accuracy_list])
            if(is_best):
                print("****New best overall accuracy: {0:.3f}****".format(average_prec1))
            save_test_results(is_best, fine_tune_test_dir, confusion_matrix, score_batch_list)
        
    print('***Smallest MSE: ' +  str(best_error) + "***")

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
        loss = criterion(output, target_var)
        
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

def validate_single(val_loader, val_loader_original, model, criterion, epoch, update_c_matrix=False, fine_tune_test_dir=None):
    """Perform validation on the validation set"""
    #Confusion matrix
    num_classes = 10
    confusion_matrix = ConfusionMeter(num_classes)
    accuracy_list = [AverageMeter() for i in range(0, num_classes)]
    score_batch_list = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    average_pos = AverageMeter()
    average_neg = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, ((input, target), (input2, target2)) in enumerate(zip(val_loader, val_loader_original)):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # confusion matrix
        if(update_c_matrix == True):
            #Obtain the original target, not 0,1
            DEST_DIR = TEST_DIR if fine_tune_test_dir is None else fine_tune_test_dir
            target_orig = target2
            score_batch_i_path = os.path.join(DEST_DIR, 'batch-'+str(i)+'-score')
            score_matrix = torch.load(score_batch_i_path)
            scored_output = 1.5*output[:,1] - output[:,0]
            score_matrix[:,args.target_label] = scored_output.data
            score_batch_list.append(score_matrix)
            
            prediction_value, prediction_idx = torch.max(score_matrix, 1)
            confusion_matrix.add(prediction_idx, target_orig)
            matched_prediction = target_orig[target_orig == prediction_idx]
            
            for idx, class_accuracy in enumerate(accuracy_list):
                class_target_num = target_orig[target_orig == idx].shape[0] if len(target_orig[target_orig == idx].shape)!=0 else 0
                class_pred_num = matched_prediction[matched_prediction == idx].shape[0] if len(matched_prediction[matched_prediction == idx].shape)!=0 else 0
                accuracy_val = class_pred_num/class_target_num if class_target_num!=0 else 0
                class_accuracy.update(accuracy_val, target_orig.shape[0])  
         
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

        if i % args.print_freq == 0 or i==len(val_loader)-1:
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

    return average_pos.avg, average_neg.avg, confusion_matrix, score_batch_list, accuracy_list

def validate_train(train_loader, model, criterion, epoch):
    """Perform validation on the training set"""
    idx_list = []
    score_list = []
    sorted_train_list = []
    # switch to evaluate mode
    model.eval()
    
    for i, (input, target) in enumerate(train_loader):
        target_cuda = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target_cuda, volatile=True)
        # compute output
        output = model(input_var)
        score = output[:,1] - output[:,0]
        score = score.data.cpu()*(target*2-1).float()
        # append the output score to the list
        idx_list += [idx+i*input.shape[0] for idx in range(input.shape[0])]
        score_list += [score[idx] for idx in range(score.shape[0])]
   
    score_idx_list = zip(score_list, idx_list) 
    sorted_train_list = [(score, idx) for score, idx in sorted(score_idx_list, key = lambda pair : pair[0])]
    
    print("Sorted list info")
    print(len(sorted_train_list))
    sorted_score_list, sorted_idx_list = zip(*sorted_train_list)
    for i in range(20):
        print(sorted_idx_list[i])
        print(sorted_score_list[i])

    return sorted_idx_list#sorted_train_list#train_data_list, train_label_list, score_list


def save_checkpoint(state, is_best, fine_tune_run_dir, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = os.path.join("runs_"+args.dataset, str(args.target_label)) if fine_tune_run_dir is None else fine_tune_run_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory ,filename)
    torch.save(state, filename)
    if is_best:
        print("Saving best model...")
        shutil.copyfile(filename, os.path.join(directory , args.name+'_model_best.pth.tar'))
        print("Best model saved success")

def save_test_results(is_best, fine_tune_dir, confusion_matrix, score_batch_list):
    if fine_tune_dir is None and is_best:
        print("Saving test batches & confusion matrix")
        c_matrix_path = os.path.join(TEST_DIR, "confusion_matrix")
        torch.save(confusion_matrix.value(), c_matrix_path)
        for idx, score_batch in enumerate(score_batch_list):
            score_batch_i_path = os.path.join(TEST_DIR, 'batch-'+str(idx)+'-score')
            torch.save(score_batch, score_batch_i_path)
    elif not fine_tune_dir is None:
        print("(fine tune)Saving test batches & confusion matrix")
        c_matrix_path = os.path.join(fine_tune_dir, "confusion_matrix")
        torch.save(confusion_matrix.value(), c_matrix_path)
        for idx, score_batch in enumerate(score_batch_list):
            score_batch_i_path = os.path.join(fine_tune_dir, 'batch-'+str(idx)+'-score')
            torch.save(score_batch, score_batch_i_path)


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

def compute_c_matrix_prec(c_matrix):
    total_test_samples = 10000
    correct_samples = 0
    for i in range(0, c_matrix.shape[0]):
        correct_samples += c_matrix[i][i]
    return correct_samples/total_test_samples

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
    #wrong_outputs = selected_outputs_normal[selected_outputs_normal <= selected_outputs_opposite]
    #print("Incorrect")
    #print(len(wrong_outputs))
    norm_mean = torch.mean(selected_outputs_normal)
    oppo_mean = torch.mean(selected_outputs_opposite)

    #if(norm_mean < -1 or norm_mean > 1 or oppo_mean < -1 or oppo_mean > 1):
    #    print("norm_mean:" + str(norm_mean) + " oppo_mean:"+str(oppo_mean))
    #print(selected_outputs_normal.shape)
    #print(selected_outputs_normal.size(0))
    
    return norm_mean-oppo_mean, selected_outputs_normal.shape[0]

if __name__ == '__main__':
    main()
