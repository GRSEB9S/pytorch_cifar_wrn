import argparse
import os
import sys
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
from wrn_selector import WideResNetSelector

from cifar_10_custom import CIFAR10

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

TEST_DIR = "test_results"

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--k-value', default=1, type=int,
                    help='getting the prediction in top k counts as correct')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--name', default='selector', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--re-test', type=bool, default=True, 
                    help='Re-test results from all models?')
def main():
    global args
    args = parser.parse_args()
    model_dir = os.path.join("runs_"+args.dataset, args.name)
    if args.tensorboard: configure(model_dir)

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    num_classes = 10 if args.dataset == 'cifar10' else 100

    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    
    model_file_name = None
    for file_name in os.listdir(model_dir):
        if(file_name.endswith("best.pth.tar")):
                model_file_name = file_name
    param = model_file_name.split('-')#dataset, layers, widen-factor, drop_rate, trailing_name
    num_layers = int(param[0])
    widen_factor = int(param[1])
    drop_rate = float(param[2])
        
    model = WideResNetSelector(num_layers, num_classes,
                            widen_factor, drop_rate)
    model = model.cuda()
    best_model_path = os.path.join(model_dir, model_file_name)
    print("=> loading selector")
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded selector success")
    else:
        print(best_model_path +" doesn't exist, loading failed")
        sys.exit()
    
    # start testing
    # create test dir for saving results
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    
    test_model(model, val_loader, num_classes)

def test_model(model, val_loader, num_classes):
    """Perform evaluation on the test set"""
    confusion_matrix = ConfusionMeter(num_classes)   
    
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()     
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        _, pred = output.data.topk(args.k_value, 1, True, True)
        
        # compute the confusion matrix & accuracy
        confusion_matrix.add(pred[:,0].view(-1), target)

        if i % args.print_freq == 0 or i == len(val_loader)-1:
            print('Test: [{0}/{1}]\t'.format(
                      i, len(val_loader)))
            print("Confusion Matrix---")
            c_matrix = confusion_matrix.value()
            print(c_matrix)
            accuracy_list = []
            for row in range(c_matrix.shape[1]):
                row_sum = torch.sum(torch.from_numpy(c_matrix[row]))
                accuracy_list.append(c_matrix[row][row]/row_sum)
            print("Class Accuracies(Average)---")
            print(accuracy_list)

    c_matrix_path = os.path.join(TEST_DIR, "confusion_matrix_selector")
    torch.save(confusion_matrix.value(), c_matrix_path)

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
