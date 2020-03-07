import torch
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def tensor_to_image(tensor):
  """
  Convert a torch tensor into a numpy ndarray for visualization.

  Inputs:
  - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

  Returns:
  - ndarr: A uint8 numpy array of shape (H, W, 3)
  """
  tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
  ndarr = tensor.to('cpu', torch.uint8).numpy()
  return ndarr


def fix_random_seed(seed_no=0):
  """
  Fix random seed to get a deterministic output

  Inputs:
  - seed_no: seed number to be fixed
  """
  torch.manual_seed(seed_no)
  torch.cuda.manual_seed(seed_no)
  random.seed(seed_no)


def visualize_dataset(X_data, y_data, samples_per_class, class_list):
  """
  Make a grid-shape image to plot

  Inputs:
  - X_data: set of [batch, 3, width, height] data
  - y_data: paired label of X_data in [batch] shape
  - samples_per_class: number of samples want to present
  - class_list: list of class names
    e.g.) ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  Outputs:
  - An grid-image that visualize samples_per_class number of samples per class
  """
  img_half_width = X_data.shape[2] // 2
  samples = []
  for y, cls in enumerate(class_list):
    plt.text(-4, (img_half_width * 2 + 2) * y + (img_half_width + 2), cls, ha='right')
    idxs = (y_data == y).nonzero().view(-1)
    for i in range(samples_per_class):
      idx = idxs[random.randrange(idxs.shape[0])].item()
      samples.append(X_data[idx])

  img = make_grid(samples, nrow=samples_per_class)
  return tensor_to_image(img)


def decode_captions(captions, idx_to_word):
    """
    Decoding caption indexes into words.
    Inputs:
    - captions: Caption indexes in a tensor of shape (Nx)T.
    - idx_to_word: Mapping from the vocab index to word.

    Outputs:
    - decoded: A sentence (or a list of N sentences).
    """
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

def smoothen(curve_list, window_size=5):
    result_list = []
    for i in range(len(curve_list)):
        if i+1 < window_size:
            avg = np.mean(curve_list[:i+1])
        else:
            avg = np.mean(curve_list[i+1-window_size:i+1])
        result_list.append(avg)

    return result_list


class Partition(object):
    """
    One partition of the dataset with interface compatible with pytorch when in distributed setting
    i.e., with support of __len__ and __getitem__ functions
    """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """
    Used to partition a dataset into several nonoverlapping partitions.
    This is normally used in distributed training when each machine wants to access a nonoverlapping partition of the entire dataset.

    Example:
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(dataset, size, rank):
    """
    Partition the dataset into size nonoverlapping chunks, and return the chunk with idx rank.
    """
    partition_sizes = [1.0 / size for _ in range(size)]
    partitioner = DataPartitioner(dataset, partition_sizes)
    partition = partitioner.use(rank)
    return partition

