import os
import torch

from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10


def _extract_tensors(dset, num=None):
  """
  Extract the data and labels from a CIFAR10 dataset object and convert them to
  tensors.

  Input:
  - dset: A torchvision.datasets.CIFAR10 object
  - num: Optional. If provided, the number of samples to keep.

  Returns:
  - x: float32 tensor of shape (N, 3, 32, 32)
  - y: int64 tensor of shape (N,)
  """
  x = torch.tensor(dset.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
  y = torch.tensor(dset.targets, dtype=torch.int64)
  if num is not None:
    if num <= 0 or num > x.shape[0]:
      raise ValueError('Invalid value num=%d; must be in the range [0, %d]'
                       % (num, x.shape[0]))
    x = x[:num].clone()
    y = y[:num].clone()
  return x, y


def cifar10_old(num_train=None, num_test=None):
  """
  Return the CIFAR10 dataset, automatically downloading it if necessary.
  This function can also subsample the dataset.

  Inputs:
  - num_train: [Optional] How many samples to keep from the training set.
    If not provided, then keep the entire training set.
  - num_test: [Optional] How many samples to keep from the test set.
    If not provided, then keep the entire test set.

  Returns:
  - x_train: float32 tensor of shape (num_train, 3, 32, 32)
  - y_train: int64 tensor of shape (num_train, 3, 32, 32)
  - x_test: float32 tensor of shape (num_test, 3, 32, 32)
  - y_test: int64 tensor of shape (num_test, 3, 32, 32)
  """
  download = not os.path.isdir('cifar-10-batches-py')
  dset_train = CIFAR10(root='.', download=download, train=True)
  dset_test = CIFAR10(root='.', train=False)
  x_train, y_train = _extract_tensors(dset_train, num_train)
  x_test, y_test = _extract_tensors(dset_test, num_test)
 
  return x_train, y_train, x_test, y_test


def mnist(root='~/data/mnist', train_batch_size=128, test_batch_size=128, train_shuffle=True, test_shuffle=False, num_workers=0):
    root = os.path.expanduser(root)  
    training_set = datasets.MNIST(root, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
        
    num_workers_dl = num_workers
    train_loader = DataLoader(training_set, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=num_workers_dl)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=test_batch_size, shuffle=test_shuffle, num_workers=num_workers_dl)
    return train_loader, test_loader

def cifar10(root='~/data/cifar10', train_batch_size=128, test_batch_size=128, train_shuffle=True, test_shuffle=False, num_workers=0):
    root = os.path.expanduser(root)
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
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
    # data prep for test set
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    # load training and test set here:
    training_set = datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=train_batch_size,
                                                  shuffle=train_shuffle, num_workers=num_workers)
    testset = datasets.CIFAR10(root=root, train=False,
                                               download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                 shuffle=test_shuffle, num_workers=num_workers)
    return train_loader, test_loader

def cifar100(root='~/data/cifar100', train_batch_size=128, test_batch_size=128, train_shuffle=True, test_shuffle=False, num_workers=0):
    root = os.path.expanduser(root)    
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
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
    # data prep for test set
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    # load training and test set here:
    training_set = datasets.CIFAR100(root=root, train=True,
                                                download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=train_batch_size,
                                                  shuffle=train_shuffle, num_workers=num_workers)
    testset = datasets.CIFAR100(root=root, train=False,
                                               download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                 shuffle=test_shuffle, num_workers=num_workers)
    return train_loader, test_loader

def svhn(root='~/data/svhn', train_batch_size=128, test_batch_size=128, train_shuffle=True, test_shuffle=False, num_workers=0):
    root = os.path.expanduser(root)
    training_set = SVHN(root, split='train', transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ]))
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=train_batch_size,
                                                  shuffle=train_shuffle, num_workers=num_workers)
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    testset = SVHN(root=root, split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                 shuffle=test_shuffle, num_workers=num_workers)
    return train_loader, test_loader

def imagenet(root='~/data/imagenet', train_batch_size=1024, test_batch_size=1024, train_shuffle=True, test_shuffle=False, num_workers=0):   
    from .imagenet import ImageNet 
    root = os.path.expanduser(root)

    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_dataset = ImageNet(
            root,
            split='train',
            download=False,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    test_dataset = ImageNet(
            root,
            split='val',
            download=False,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    num_workers_dl = num_workers
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                  shuffle=train_shuffle, num_workers=num_workers_dl)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                                 shuffle=test_shuffle, num_workers=num_workers_dl)
    return train_loader, test_loader
