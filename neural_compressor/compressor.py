import os
import numpy as np
import random
import torch
import tempfile
from neural_compressor.utils.torch.rand import *
from torch.utils.data import Dataset, DataLoader, Subset
from neural_compressor.discretization import *
from torchvision import datasets, transforms

class ToInt:
    def __call__(self, pic):
        return pic * 255


class MNIST_data(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets  # useless in compression

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        target_sample = self.targets[idx]
        return data_sample, target_sample

    def __len__(self):
        return len(self.data)


class ANS:
    def __init__(self, pmfs, bits=31, quantbits=8):
        self.device = pmfs.device
        self.bits = bits
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << bits) - 1

        # normalization constants
        self.lbound = 1 << 32
        self.tail_bits = (1 << 32) - 1

        self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        # add remnant to the maximum value of the probabilites
        self.pmfs[torch.arange(0, self.seq_len),torch.argmax(self.pmfs, dim=1)] += ((1 << self.bits) - self.pmfs.sum(1))

        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=1) # compute CDF (scaled up to 2**n)
        self.cdfs = torch.cat([torch.zeros([self.cdfs.shape[0], 1], dtype=torch.long, device=self.device), self.cdfs], dim=1) # pad with 0 at the beginning

        # move cdf's and pmf's the cpu for faster encoding and decoding
        self.cdfs = self.cdfs.cpu().numpy()
        self.pmfs = self.pmfs.cpu().numpy()

        assert self.cdfs.shape == (self.seq_len, self.support + 1)
        assert np.all(self.cdfs[:,-1] == (1 << bits))

    def encode(self, x, symbols):
        for i, s in enumerate(symbols):
            pmf = int(self.pmfs[i,s])
            if x[-1] >= ((self.lbound >> self.bits) << 32) * pmf:
                x.append(x[-1] >> 32)
                x[-2] = x[-2] & self.tail_bits
            x[-1] = ((x[-1] // pmf) << self.bits) + (x[-1] % pmf) + int(self.cdfs[i, s])
        return x

    def decode(self, x):
        sequence = np.zeros((self.seq_len,), dtype=np.int64)
        for i in reversed(range(self.seq_len)):
            masked_x = x[-1] & self.mask
            s = np.searchsorted(self.cdfs[i,:-1], masked_x, 'right') - 1
            sequence[i] = s
            x[-1] = int(self.pmfs[i,s]) * (x[-1] >> self.bits) + masked_x - int(self.cdfs[i, s])
            if x[-1] < self.lbound:
                x[-1] = (x[-1] << 32) | x.pop(-2)
        sequence = torch.from_numpy(sequence).to(self.device)
        return x, sequence


# Compress function
def compress_online(model, x, dataset, quantbits=10, nz=2, bitswap=False, array_dir='tmp', save_fn='tmp_compressed_fn'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dataset == 'cifar':
        zdim = 8 * 16 * 16
        xdim = 32 ** 2 * 3
    elif dataset == 'mnist' or dataset == 'fashionmnist':
        zdim = 1 * 16 * 16
        xdim = 32 ** 2 * 1
    else:
        raise ValueError("Dataset not support.")
    zrange = torch.arange(zdim)
    xrange = torch.arange(xdim)
    ansbits = 31
    type = torch.float64

    # For replication
    np.random.seed(100)
    random.seed(50)
    torch.manual_seed(50)
    torch.cuda.manual_seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Discretization
    zendpoints, zcentres = discretize(nz, quantbits, type, device, model, dataset)
    xbins = ImageBins(type, device, xdim)
    xendpoints = xbins.endpoints()
    xcentres = xbins.centres()

    # Compression
    model.compress()
    state = list(map(int, np.random.randint(low=1 << 16, high=(1 << 32) - 1, size=10000,
                                            dtype=np.uint32)))  # fill state list with 'random' bits
    state[-1] = state[-1] << 32
    initialstate = state.copy()
    restbits = None

    # ===== Alice (sender) =====
    xi = 0
    x = x.to(device).view(xdim)

    if bitswap:
        #  ===== Bit-Swap =====
        # inference and generative model
        for zi in range(nz):
            # inference model
            input = zcentres[zi - 1, zrange, zsym] if zi > 0 else xcentres[xrange, x.long()]
            mu, scale = model.infer(zi)(given=input)
            cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t()  # most expensive calculation?
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

            # decode z
            state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)

            # save excess bits for calculations
            if xi == zi == 0:
                restbits = state.copy()
                assert len(
                    restbits) > 1, "too few initial bits"  # otherwise initial state consists of too few bits

            # generative model
            z = zcentres[zi, zrange, zsymtop]
            mu, scale = model.generate(zi)(given=z)
            cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu,
                                scale).t()  # most expensive calculation?
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

            # encode z or x
            state = ANS(pmfs, bits=ansbits, quantbits=(quantbits if zi > 0 else 8)).encode(state,
                                                                                           zsym if zi > 0 else x.long())

            zsym = zsymtop
    else:
        # ===== BB-ANS =====
        # inference and generative model
        zs = []
        for zi in range(nz):
            # inference model
            input = zcentres[zi - 1, zrange, zsym] if zi > 0 else xcentres[xrange, x.long()]
            mu, scale = model.infer(zi)(given=input)
            cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t()  # most expensive calculation?
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

            # decode z
            state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)
            zs.append(zsymtop)

            zsym = zsymtop

        # save excess bits for calculations
        if xi == 0:
            restbits = state.copy()
            assert len(restbits) > 1  # otherwise initial state consists of too few bits

        for zi in range(nz):
            # generative model
            zsymtop = zs.pop(0)
            z = zcentres[zi, zrange, zsymtop]
            mu, scale = model.generate(zi)(given=z)
            cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu,
                                scale).t()  # most expensive calculation?
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

            # encode z or x
            state = ANS(pmfs, bits=ansbits, quantbits=(quantbits if zi > 0 else 8)).encode(state,
                                                                                           zsym if zi > 0 else x.long())

            zsym = zsymtop

        assert zs == []

    # prior
    cdfs = logistic_cdf(zendpoints[-1].t(), torch.zeros(1, device=device, dtype=type),
                        torch.ones(1, device=device, dtype=type)).t()
    pmfs = cdfs[:, 1:] - cdfs[:, :-1]
    pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

    # encode prior
    state = ANS(pmfs, bits=ansbits, quantbits=quantbits).encode(state, zsymtop)
    # print("final state:", len(state))

    # calculating bits
    del state[0:10000]
    state_array = np.array(state, dtype=np.uint32)
    np.save(os.path.join(array_dir, save_fn), state_array)
    size_state = os.path.getsize(os.path.join(array_dir, save_fn + '.npy')) * 8
    os.remove(os.path.join(array_dir, save_fn + '.npy'))
    return size_state, state_array



