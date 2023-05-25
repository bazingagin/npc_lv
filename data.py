import csv
import numpy as np
import torch
from collections import defaultdict
from torchvision import datasets, transforms

from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader, Subset


class ToInt:
    def __call__(self, pic):
        return pic * 255


def read_fn_label(fn):
    text2label = {}
    with open(fn) as fo:
        reader = csv.reader(fo, delimiter=',', quotechar='"')
        for row in reader:
            label, title, desc = row[0], row[1], row[2]
            text = '. '.join([title, desc])
            text2label[text] = label
    return text2label

def read_label(fn):
    labels = []
    with open(fn) as fo:
        reader = csv.reader(fo, delimiter=',', quotechar='"')
        for row in reader:
            label, title, desc = row[0], row[1], row[2]
            labels.append(label)
    return labels

def read_torch_text_labels(ds, indicies):
    text_list = []
    label_list = []
    for i, (label, line) in enumerate(ds):
        if i in indicies:
            text_list.append(line)
            label_list.append(label)
    return text_list, label_list

def read_img_with_label(dataset, indicies, flatten=True):
    imgs = []
    labels = []
    for idx in indicies:
        img = np.array(dataset[idx][0])
        label = dataset[idx][1]
        if flatten:
            img = img.flatten()
        imgs.append(img)
        labels.append(label)
    return np.array(imgs), np.array(labels)

def read_img_label(dataset, indicies):
    labels = []
    for idx in indicies:
        label = dataset[idx][1]
        labels.append(label)
    return labels

def pick_n_sample_from_each_class(fn, n, idx_only=False):
    label2text = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []
    with open(fn) as fo:
        reader = csv.reader(fo, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            label, title, desc = row[0], row[1], row[2]
            text = '. '.join([title, desc])
            label2text[label].append(text)
            label2idx[label].append(i)
        for cl in label2text:
            class2count[cl] = len(label2text[cl])
    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n, replace=False)
        select_text = np.array(label2text[c])[select_idx]
        select_text_idx = np.array(label2idx[c])[select_idx]
        recorded_idx += list(select_text_idx)
        result+=list(select_text)
        labels+=[c]*n
    print(len(result))
    if idx_only:
        return recorded_idx
    else:
        return result, labels

def pick_n_sample_from_each_class_given_dataset(ds, n, output_fn):
    label2text = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []
    for i, (label, text) in enumerate(ds):
        label2text[label].append(text)
        label2idx[label].append(i)
    for cl in label2text:
        class2count[cl] = len(label2text[cl])
    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n, replace=False)
        select_text = np.array(label2text[c])[select_idx]
        select_text_idx = np.array(label2idx[c])[select_idx]
        recorded_idx+=list(select_text_idx)
        result+=list(select_text)
        labels+=[c]*n
    print(len(result))
    np.save(output_fn, np.array(recorded_idx))
    return result, labels


def pick_n_sample_from_each_class_img(dataset, n, flatten=False):
    label2img = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = [] #for replication
    for i,pair in enumerate(dataset):
        img, label = pair
        if type(label) != int:
            label = label.item()
        if flatten:
            img = np.array(img).flatten()
        label2img[label].append(np.array(img))
        label2idx[label].append(i)
    for cl in label2img:
        class2count[cl] = len(label2img[cl])
    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n, replace=False)
        select_img = np.array(label2img[c])[select_idx]
        select_img_idx = np.array(label2idx[c])[select_idx]
        recorded_idx+=list(select_img_idx)
        result+=list(select_img)
        labels+=[c]*n
    return result, labels, recorded_idx



