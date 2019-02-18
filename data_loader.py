from __future__ import print_function
from torchvision import datasets, transforms
import torch
from AdvMNIST import AdvMNIST
import os
import numpy as np


def load_data(batch_size, source_label, data_set):

    if 'MNIST' == data_set:
        train_loader, test_loader = load_mnist(batch_size, source_label)
    elif 'CIFAR10' == data_set:
        train_loader, test_loader = load_cifar10(batch_size, source_label)

    return train_loader, test_loader


def load_mnist(batch_size, source_label):

    print('==> Preparing data..')

    # Training dataset
    dataset_train = datasets.MNIST(root='.', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    if source_label is not None:
        idx = dataset_train.train_labels == source_label
        dataset_train.train_labels = dataset_train.train_labels[idx]
        dataset_train.train_data = dataset_train.train_data[idx]

    train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True)#, num_workers=4)

    # Test dataset
    dataset_test = datasets.MNIST(root='.', train=False, download=True,
                                  transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    if source_label is not None:
        idx = dataset_test.test_labels == source_label
        dataset_test.test_labels = dataset_test.test_labels[idx]
        dataset_test.test_data = dataset_test.test_data[idx]

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True)#, num_workers=4)

    return train_loader, test_loader


def load_cifar10(batch_size, source_label):
    print('==> Preparing data..')

    # Training dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform_train)

    if source_label is not None:
        train_labels_np = np.array(train_set.train_labels)
        idx = np.where(train_labels_np == source_label)

    train_idx = idx[0]
    print(train_idx.shape)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=train_sampler)# num_workers=2)

    # Tasting dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    print(test_set.test_data.shape)

    if source_label is not None:
        test_labels_np = np.array(test_set.test_labels)
        idx = np.where(test_labels_np == source_label)

    test_idx = idx[0]
    print(test_idx.shape)

    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=test_sampler)

    return train_loader, test_loader


# loader for AdvMnist
def load_adv_data(dataset_folder, batch_size, source_label=None):

    dataset_validation = AdvMNIST(csv_path=os.path.join(dataset_folder, 'adversarial_mnist_csv.csv'),
                                  transform=transforms.Compose([
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    if source_label is not None:
        idx = dataset_validation.Original_arr == source_label
        dataset_validation.Original_arr = dataset_validation.Original_arr[idx]
        dataset_validation.image_arr = dataset_validation.image_arr[idx]
        dataset_validation.Advlabel_arr = dataset_validation.Advlabel_arr[idx]

    validation_loader = torch.utils.data.DataLoader(
        dataset_validation, batch_size=batch_size, shuffle=True, num_workers=4)

    return validation_loader
