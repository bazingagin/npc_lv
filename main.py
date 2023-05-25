import argparse
from experiments import *
from compressors import *
from utils import *
from data import *


class ToInt:
    def __call__(self, pic):
        return pic * 255


class Permute:
    def __call__(self, pic):
        return pic.permute(1,2,0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', help='Choices are: `mnist`, `fashionmnist`, `cifar`')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--shot', default=10)
    parser.add_argument('--compressor', default='gzip')
    parser.add_argument('--distance', default='NCD')
    parser.add_argument('--online', default=False, action='store_true')
    parser.add_argument('--replicate', default=False, action='store_true')
    parser.add_argument('--c_train_dir', help='Directory includes compressed training files')
    parser.add_argument('--c_test_dir', help='Directory includes compressed test files')
    parser.add_argument('--c_combined_dir', help='Directory includes compressed aggregated files')

    args = parser.parse_args()

    if args.dataset == 'mnist':
        transform_ops = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
        train_dataset = datasets.MNIST(root=args.data_dir, train=True, transform=transform_ops, download=True)
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, transform=transform_ops, download=True)
        k = 2
    elif args.dataset == 'fashionmnist':
        transform_ops = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
        train_dataset = datasets.FashionMNIST(root=args.data_dir, train=True, transform=transform_ops, download=True)
        test_dataset = datasets.FashionMNIST(root=args.data_dir, train=False, transform=transform_ops, download=True)
        k = 2
    elif args.dataset == 'cifar':
        if args.online and args.compressor in ['bbans', 'bb-ans', 'bitswap', 'bit-swap']:
            transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
        else:
            transform_ops = transforms.Compose([transforms.ToTensor(), ToInt(), Permute()])
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform_ops, download=True)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_ops, download=True)
        k = 3
    else:
        raise ValueError("Dataset not supported.")

    if args.compressor in ['bbans', 'bb-ans', 'bitswap', 'bit-swap']:
        if args.online:
            compressor = BitSwapOnlineCompressor(args.dataset, 2, args.compressor in ['bitswap', 'bit-swap'])
            compressed_provided = False
        else:
            assert args.c_train_dir is not None and args.c_test_dir is not None and args.c_combined_dir is not None, \
                "Compressed files dir need to be provided. Otherwise, use --online option. "
            compressor = BitSwapCompressor(args.c_train_dir, args.c_test_dir, args.c_combined_dir)
            compressed_provided = True
    else:
        compressor = ImageCompressor(args.compressor)
        compressed_provided = False

    knn_exp_img = KnnExpImg(agg_by_avg, compressor, eval(args.distance), args.dataset, args.data_dir, transform_ops)
    shot_num = int(args.shot)

    # For replication
    train_idx_fn = 'input/{}/trainperclass100_idx.npy'.format(args.dataset)
    test_idx_fn = 'input/{}/testperclass100_idx.npy'.format(args.dataset)
    if args.replicate and os.path.exists(train_idx_fn) and os.path.exists(test_idx_fn):
        # Assume index files are provided and compressed files are provided
        train_idx = np.load(train_idx_fn)
        train_labels = read_img_label(train_dataset, train_idx)
        test_idx = np.load(test_idx_fn)
        test_labels = read_img_label(test_dataset, test_idx)
        num_per_class = 100
        if shot_num < 100:
            few_shot_train_argidx = np.array([], dtype=np.uint8)
            few_shot_train_idx = np.array([], dtype=np.uint8)
            for i in range(0, len(train_idx), num_per_class):
                selected_train_argidx = np.random.choice(range(i, i + num_per_class), shot_num, replace=False)
                selected_train_idx = train_idx[selected_train_argidx]
                few_shot_train_argidx = np.append(few_shot_train_argidx, selected_train_argidx)
                few_shot_train_idx = np.append(few_shot_train_idx, selected_train_idx)
            few_shot_train_label = read_img_label(train_dataset, few_shot_train_idx)
            knn_exp_img.calc_dis(test_idx, few_shot_train_idx, compressed_provided)
            knn_exp_img.calc_acc(k, test_labels, train_label=few_shot_train_label)
        elif shot_num == 100:
            knn_exp_img.calc_dis(test_idx, train_idx, compressed_provided)
            knn_exp_img.calc_acc(k, test_labels, train_label=train_labels)
        else:
            raise ValueError()
    # For more "freestyle
    else:
        # No compressed files are provided
        test_data, test_labels, test_idx = pick_n_sample_from_each_class_img(test_dataset, 10)
        few_shot_train_data, few_shot_train_label, few_shot_train_idx = pick_n_sample_from_each_class_img(train_dataset, shot_num)
        knn_exp_img.calc_dis(test_idx, few_shot_train_idx, compressed_provided)
        knn_exp_img.calc_acc(k, test_labels, train_label=few_shot_train_label)