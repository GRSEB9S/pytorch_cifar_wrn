from __future__ import print_function
from __future__ import division
from PIL import Image
import os
import os.path
import errno
import torch
import numpy as np
import sys
from torchnet.meter import ConfusionMeter
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
#from .utils import download_url, check_integrity


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, target_label, root, train=True,
                 transform=None, target_transform=None,
                 download=False, confusion_matrix=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.target_label = target_label
        self.confusion_matrix = confusion_matrix
        assert 0 <= self.target_label <= 9
        #if download:
        #    print("Cannot download on custom class, please download another way")
            #self.download()
        """
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        """
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            
            is_selective = False if self.confusion_matrix is None else True
            #Iterate through data batches 
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                #print(file)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                
                batch_train_data, batch_train_label = self.extract_data(entry, self.target_label, balanced= not is_selective)
                self.train_data.append(batch_train_data)
                self.train_labels += batch_train_label
                #self.select_target(batch_train_label, self.target_label)

                """
                self.train_data.append(entry['data'])
                
                if 'labels' in entry:
                    #print(type(entry['labels']))
                    self.train_labels += self.select_target(entry['labels'], self.target_label)
                else:
                    #print(type(entry['fine_labels']))
                    self.train_labels += self.select_target(entry['fine_labels'], self.target_label)
                fo.close()
                """
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            if is_selective:
                class_ratio_list = self.get_class_ratio_list(target_label, self.confusion_matrix)
                self.train_data, self.train_labels = self.selected_data(self.train_data, self.train_labels, class_ratio_list)
                print("training class ratios: ")
                print(class_ratio_list)
                print("training data shape: " + str(self.train_data.shape))
                #print(len(self.train_labels))
            self.train_labels = self.select_target(self.train_labels, self.target_label)

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            
            self.test_data = entry['data']
            
            if 'labels' in entry:
                self.test_labels = self.select_target(entry['labels'], self.target_label)
            else:
                self.test_labels = self.select_target(entry['fine_labels'], self.target_label)
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    """
    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
    """
    def extract_data(self, entry, target_label, balanced):
        data = entry['data'].reshape(-1,3072)
        label = entry['labels']
        result_data = []
        result_label = []
        for idx, val in enumerate(label):
            if(val == target_label):
                #Increase target sample amount
                if balanced:
                    for i in range(9):
                        result_data.append(data[idx])
                        result_label.append(val)
                else:
                    for i in range(1):
                        result_data.append(data[idx])
                        result_label.append(val)
            else:
                result_data.append(data[idx])
                result_label.append(val)
        result_data = np.asarray(result_data).reshape(-1,3072)
        #print(result_data.shape[0])
        #print(len(result_label))
        return result_data, result_label
    
    def get_class_ratio_list(self, target_label, confusion_matrix):
        if confusion_matrix is None:
            return []
        #Obtain the class proportions on misclassified samples
        class_column = confusion_matrix[:, target_label]
        misclassified_sum = np.sum(class_column[:target_label])
        misclassified_sum += np.sum(class_column[target_label+1:]) if target_label!=9 else 0
        class_ratio_list = []
        for i in range(10):
            if(i == target_label):
                class_ratio_list.append(1.0)
            else:
                class_ratio_list.append(class_column[i]/misclassified_sum)
        return class_ratio_list

    def selected_data(self, train_data, train_labels, class_ratio_list):
        result_data = []
        result_label = []
       
        #Create dataset
        for idx, val in enumerate(train_labels):
            if(np.random.uniform() <= class_ratio_list[val]):
                result_data.append(train_data[idx])
                result_label.append(val)
        
        result_data = np.asarray(result_data).reshape((-1,)+train_data.shape[1:])
        #print(result_data.shape[0])
        #print(len(result_label))
        return result_data, result_label
        

    def select_target(self, label_list, target_label):
        for idx, val in enumerate(label_list):
            if(label_list[idx] == target_label):
                label_list[idx] = 1
            else:
                label_list[idx] = 0
        return label_list

class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
]
    
