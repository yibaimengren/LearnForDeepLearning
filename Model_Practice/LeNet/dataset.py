import os
import struct
import numpy as np
from torch.utils.data import Dataset

class Mnist(Dataset):
    def __init__(self, train=True, transforms=None):
        super(Mnist, self).__init__()
        if train:
            self.images, self.labels = load_mnist(path='MNIST', kind='train')
        else:
            self.images, self.labels = load_mnist(path='MNIST', kind='test')
        self.trans = transforms

    def __getitem__(self, index):
        img = self.images[index].reshape(28, 28)
        if self.trans:
            img = self.trans(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path,f'{kind}-images.idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels



# import matplotlib.pyplot as plt
# dataset = Mnist()
# img, label = dataset[9]
# plt.title(str(label))
# plt.imshow(img)
# plt.show()
