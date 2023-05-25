# Experiment framework
import os
import torch
import numpy as np
import statistics
import operator
from collections import Counter, defaultdict
from tqdm import tqdm
from functools import partial
from itertools import repeat
from torchvision import datasets, transforms
from copy import deepcopy
from statistics import mode
import pickle


class KnnExpImg:
    def __init__(self, agg_f, comp, dis, ds_name, ds_dir, trans_ops):
        self.aggregation_func = agg_f
        self.compressor = comp
        self.distance_func = dis
        self.dis_matrix = []
        if ds_name == 'mnist':
            self.train_dataset = datasets.MNIST(root=ds_dir, train=True, transform=trans_ops, download=True)
            self.test_dataset = datasets.MNIST(root=ds_dir, train=False, transform=trans_ops, download=True)
        elif ds_name == 'cifar':
            self.train_dataset = datasets.CIFAR10(root=ds_dir, train=True, transform=trans_ops, download=True)
            self.test_dataset = datasets.CIFAR10(root=ds_dir, train=False, transform=trans_ops, download=True)
        elif ds_name == 'fashionmnist':
            self.train_dataset = datasets.FashionMNIST(root=ds_dir, train=True, transform=trans_ops, download=True)
            self.test_dataset = datasets.FashionMNIST(root=ds_dir, train=False, transform=trans_ops, download=True)

    def calc_dis(self, test_indicies, train_indicies, compressed_provided=False):
        for test_idx in tqdm(test_indicies):
            if compressed_provided:
                test_img_len = self.compressor.get_compressed_len(test_idx, 'test')
            else:
                test_img_len = self.compressor.get_compressed_len(self.test_dataset[test_idx][0])
            distance4i = []
            for train_idx in train_indicies:
                if compressed_provided:
                    train_img_len = self.compressor.get_compressed_len(train_idx, 'train')
                    combined_img_len = self.compressor.get_compressed_len(test_idx, train_idx)
                else:
                    test_img = self.test_dataset[test_idx][0]
                    train_img = self.train_dataset[train_idx][0]
                    train_img_len = self.compressor.get_compressed_len(train_img)
                    combined_img_len = self.compressor.get_compressed_len(self.aggregation_func(test_img.squeeze(), train_img.squeeze()))
                distance = self.distance_func(test_img_len, train_img_len, combined_img_len)
                distance4i.append(distance)
            self.dis_matrix.append(distance4i)

    def calc_dis_with_latent(self, data, train_data):
        for i, p1 in tqdm(enumerate(data)):
            distance4i = []
            for j, p2 in enumerate(train_data):
                distance = self.distance_func(p1, p2)
                distance4i.append(distance)
            self.dis_matrix.append(distance4i)

    def calc_acc(self, k, label, train_label=None, provided_distance_matrix=None):
        if provided_distance_matrix is not None:
            self.dis_matrix = provided_distance_matrix
        correct = []
        pred = []
        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k+1
        for i in range(len(self.dis_matrix)):
            sorted_idx = np.argsort(np.array(self.dis_matrix[i]))
            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1
            sorted_pred_lab = sorted(pred_labels.items(), key=operator.itemgetter(1), reverse=True)
            most_count = sorted_pred_lab[0][1]
            if_right = 0
            most_label = sorted_pred_lab[0][0]
            for pair in sorted_pred_lab:
                if pair[1] < most_count:
                    break
                if pair[0] == label[i]:
                    if_right = 1
                    most_label = pair[0]
            pred.append(most_label)
            correct.append(if_right)
        print("Accuracy is {}".format(sum(correct)/len(correct)))
        return pred, correct