import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

import os
from PIL import Image
import torchvision.transforms as transforms
import argparse

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    print('batch numbers: %d' % int(N/batch_size))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
        # print('step: %d' % i)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def parse_args():
    parser = argparse.ArgumentParser(description='get Vaild image dir')
    parser.add_argument('--dir', dest='img_dir',
                        help='optional image dir',
                        default='None', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, imgspath):
            self.imgspath = imgspath
            self.transform = transforms.Compose([
                                 transforms.Resize(299),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.namelist = os.listdir(self.imgspath)

        def __getitem__(self, index):
            imgname = self.namelist[index]
            img_path = self.imgspath + imgname
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img

        def __len__(self):
            return len(self.namelist)\

    args = parse_args()
    img_dir = args.img_dir

    print('load images from ' + img_dir)
    imgs = IgnoreLabelDataset(img_dir)

    print("Calculating Inception Score...")
    torch.cuda.set_device(0)
    print(inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=10))
    print(img_dir)