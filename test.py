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
from wideresnet import WideResNet

from cifar_10_custom import CIFAR10

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

TEST_DIR = "test_results"

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--target-class-list', type=str,
                    help='The selected target classes to test(example: 0,1,2,3)')
parser.add_argument('--retrained-class-list', default='None', type=str,
                    help='The newly trained classes to test(example: 1,2), other tested samples are resumed')
parser.add_argument('--re-test', type=bool, default=True, 
                    help='Re-test results from all models?')
def main():
    global args
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))

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
    
    # load models
    model_list = []
    model_i_list = args.target_class_list.split(',')
    target_list = list(map(int, model_i_list))
    model_name_list = [model_i for model_i in model_i_list]#example: runs_cifar10/0/
    best_error_list = []
    root = "runs_"+args.dataset
    #print(model_name_list)
    #print(target_list)
    
    for idx, model_name in enumerate(model_name_list):
        model_dir = os.path.join(root, model_name)
        model_file_name = None
        for file_name in os.listdir(model_dir):
            if(file_name.endswith("best.pth.tar")):
                model_file_name = file_name
        param = model_file_name.split('-')#dataset, layers, widen-factor, drop_rate, trailing_name
        num_layers = int(param[0])
        widen_factor = int(param[1])
        drop_rate = float(param[2])
        
        model = WideResNet(num_layers, args.dataset == 'cifar10' and 2 or 2,
                                widen_factor, drop_rate)
        best_model_path = os.path.join(model_dir, model_file_name)
        print("=> loading model '({})'".format(model_i_list[idx]))
        if os.path.isfile(best_model_path):
            checkpoint = torch.load(best_model_path)
            best_error_list.append(checkpoint['best_error'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded model '({})' success".format(model_i_list[idx]))
            model_list.append(model)
        else:
            print(check_point_path +" doesn't exist, loading failed")
            sys.exit()
    
    # start testing
    # create test dir for saving results
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    # old model test results are not reproduced
    retrained_model_list = None
    if args.retrained_class_list != 'None':
        retrained_model_i_list = args.retrained_class_list.split(',') 
        retrained_model_list = list(map(int, retrained_model_i_list))
    test_models(model_list, retrained_model_list, val_loader, num_classes)

def test_models(model_list, retrained_model_list, val_loader, num_classes):
    """Perform evaluation on the test set"""
    confusion_matrix = ConfusionMeter(num_classes)   
    accuracy_list = [AverageMeter() for i in range(0, num_classes)]

    batch_accuracy = AverageMeter()

    batch_time = AverageMeter()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        #Load existing scores
        score_batch_i_path = os.path.join(TEST_DIR, 'batch-'+str(i)+'-score')
        score_matrix = None
        if not os.path.isfile(score_batch_i_path):
            score_matrix = torch.zeros((input.shape[0], num_classes))#shape=128*10
            score_matrix.fill_(-100)
        else:
            score_matrix = torch.load(score_batch_i_path)
            #score_matrix[:,[idx for idx in range(0, num_classes) if idx not in model_list]] = -2.0
        #print("input shape:" + str(input.shape))
        for idx, model in enumerate(model_list):
            if(args.re_test and (retrained_model_list == None or idx in retrained_model_list)):
                model.eval()
                    
                input_var = torch.autograd.Variable(input, volatile=True)
                target_var = torch.autograd.Variable(target, volatile=True)

                # compute output
                output = model(input_var)#128x2

                # measure activation level and record loss:
                scored_output = output[:,1] - output[:,0]#128*1
                score_matrix[:,idx] = scored_output.data
            #else:
            #    print("Model ({0}) test results already computed".format(idx))
        # Save computed score matrix for this batch
        torch.save(score_matrix, score_batch_i_path)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # compute the prediction
        prediction_value, prediction_idx = torch.max(score_matrix, 1)
        # compute the confusion matrix & accuracy
        confusion_matrix.add(prediction_idx, target)
        matched_prediction = target[target == prediction_idx]
        batch_accuracy.update(matched_prediction.shape[0]/target.shape[0])
        for idx, class_accuracy in enumerate(accuracy_list):
            class_target_num = target[target == idx].shape[0] if len(target[target == idx].shape)!=0 else 0
            class_pred_num = matched_prediction[matched_prediction == idx].shape[0] if len(matched_prediction[matched_prediction == idx].shape)!=0 else 0
            accuracy_val = class_pred_num/class_target_num if class_target_num!=0 else 0
            class_accuracy.update(accuracy_val, target.shape[0])

        if i % args.print_freq == 0 or i == len(val_loader)-1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\it'
                  'batch_accuracy {batch_accuracy.val:.3f} ({batch_accuracy.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, batch_accuracy=batch_accuracy))
            print("Confusion Matrix---")
            print(confusion_matrix.value())
            print("Class Accuracies(this batch)---")
            print([class_accuracy.val for class_accuracy in accuracy_list])
            print("Class Accuracies(Average)---")
            print([class_accuracy.avg for class_accuracy in accuracy_list])

    c_matrix_path = os.path.join(TEST_DIR, "confusion_matrix")
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
