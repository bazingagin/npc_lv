# Compressor Framework
import gzip
import bz2
import lzma
from PIL.PngImagePlugin import getchunks
from PIL import Image
import sys
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import io

from neural_compressor.model.vae_model import Model
from neural_compressor.compressor import compress_online


class ImageCompressor:
    def __init__(self, compressor=None, form=None, greyscale=False):
        self.form = form
        if compressor == 'gzip':
            self.compressor = gzip
        elif compressor == 'bz2':
            self.compressor = bz2
        elif compressor == 'lzma':
            self.compressor = lzma
        elif compressor == 'WebP' or compressor == 'webp':
            self.form = 'WebP'
        elif compressor == 'PNG' or compressor == 'png':
            self.form = 'PNG'
        self.greyscale = greyscale

    def compress_func(self, x):
        compressed_data = bytearray()
        # img = x.tobytes()
        if self.form == 'PNG':
            img = Image.fromarray(x.astype(np.uint8))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=self.form, optimize=True)
            compressed_data.extend(img_bytes.getvalue())
        elif self.form == 'WebP':
            img = Image.fromarray(x.astype(np.uint8))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=self.form, lossless=True, quality=100)
            compressed_data.extend(img_bytes.getvalue())
        else:
            compressed_data = self.compressor.compress(x.astype(np.uint8).tobytes())
        return compressed_data

    def get_compressed_len(self, x):
        if self.greyscale:
            greyscale_x = Image.fromarray(x.numpy().astype(np.uint8)).convert('L')
            x = np.tile(np.array(greyscale_x), [3,1,1])
            x = np.transpose(x, axes=[1,2,0])
        else:
            x = x.squeeze().cpu().detach().numpy()
        return len(self.compress_func(x))


class BitSwapCompressor:
    """
    Compressor class for both BB-ANS and Bit-Swap
    """
    def __init__(self, train_dir, test_dir, comb_dir, postfix='.png', bitsarray=False):
        self.train_compressed_dir = train_dir
        self.test_compressed_dir = test_dir
        self.comb_compressed_dir = comb_dir
        self.postfix = postfix
        self.bitsarray = bitsarray

    @staticmethod
    def accumulate_bit_length(state_array):
        bitlength = 0
        for state in state_array:
            bitlength += int(state).bit_length()
        return bitlength

    def get_compressed_len(self, img_idx, train_or_test):
        if type(train_or_test) != str:
            i1, i2 = img_idx, train_or_test
            if not self.bitsarray:
                img_fn = os.path.join(self.comb_compressed_dir, 'img_{}_{}{}.npy'.format(i1, i2, self.postfix))
                compressed_len = os.path.getsize(img_fn)
            else:
                compressed_len = self.comb_compressed_dir[img_idx*100+train_or_test]
        else:
            if train_or_test == 'train':
                if not self.bitsarray:
                    img_fn = os.path.join(self.train_compressed_dir, 'img_{}{}.npy'.format(img_idx, self.postfix))
                    compressed_len = os.path.getsize(img_fn)
                else:
                    compressed_len = self.train_compressed_dir[img_idx]
            else:
                if not self.bitsarray:
                    img_fn = os.path.join(self.test_compressed_dir, 'img_{}{}.npy'.format(img_idx, self.postfix))
                    compressed_len = os.path.getsize(img_fn)
                else:
                    compressed_len = self.test_compressed_dir[img_idx]
        return compressed_len


class BitSwapOnlineCompressor:
    def __init__(self, dataset, nz, bitswap):
        self.dataset = dataset
        self.nz = nz
        self.bitswap = bitswap
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Initialize model here to avoid IO problem
        if dataset == 'cifar':
            if nz == 8:
                reswidth = 252
            elif nz == 4:
                reswidth = 254
            elif nz == 2:
                reswidth = 255
            else:
                reswidth = 256
            assert nz > 0
            self.model = Model(xs=(3, 32, 32), nz=nz, zchannels=8, nprocessing=4, kernel_size=3, resdepth=8,
                          reswidth=reswidth).to(self.device)
            self.model.load_state_dict(
                torch.load(f'neural_compressor/model/params/cifar/nz{nz}',
                           map_location=lambda storage, location: storage
                           )
            )

        elif dataset == 'mnist' or dataset == 'fashionmnist':
            if nz == 8:
                reswidth = 61
            elif nz == 4:
                reswidth = 62
            elif nz == 2:
                reswidth = 63
            else:
                reswidth = 64
            assert nz > 0
            self.model = Model(xs=(1, 32, 32), nz=nz, zchannels=1, nprocessing=4, kernel_size=3, resdepth=8,
                          reswidth=reswidth).to(self.device)
            self.model.load_state_dict(
                torch.load(f'neural_compressor/model/params/{dataset}/nz{nz}',
                           map_location=lambda storage, location: storage
                           )
            )

        else:
            raise ValueError("Dataset not support.")
        self.model.eval()

    def compress(self, x):
        compressed_len, state_array = compress_online(self.model, x, self.dataset, nz=self.nz, bitswap=self.bitswap)
        return compressed_len, state_array

    def get_compressed_len(self, x):
        compressed_len, state_array = self.compress(x)
        return compressed_len