from __future__ import print_function

from config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import CNN_ENCODER

import os
import sys
import time
import random
import argparse
import numpy as np
import pprint

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--valid_dir', dest='Vdir', type=str, default='')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def build_models():
    Inception_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    Inception_encoder.init_trainable_weights()
    labels = Variable(torch.LongTensor(range(batch_size)))

    if cfg.TRAIN.NET_I != '':
        Iname = cfg.TRAIN.NET_I
        state_dict = torch.load(Iname)
        Inception_encoder.load_state_dict(state_dict)
        print('Load ', Iname)
    if cfg.CUDA:
        Inception_encoder = Inception_encoder.cuda()
        Inception_encoder = Inception_encoder.eval()
        labels = labels.cuda()

    return Inception_encoder, labels


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True
    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    valid_dir = args.Vdir
    dataset_val = TextDataset(cfg.DATA_DIR, 'test', valid_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # calculate R precision ##############################################################
    Inception_encoder,labels = build_models()

    netT = torch.load(cfg.TRAIN.NET_T)
    text_encoder = torch.load(cfg.TRAIN.NET_E)
    netT = netT.cuda()
    text_encoder = text_encoder.cuda()

    inception_model = Inception_encoder
    rnn_model = text_encoder
    count = len(dataloader_val)
    start_time = time.time()
    R_GT = range(100)
    P_rates = []
    for step, data in enumerate(dataloader_val, 0):
        imgs, captions, cap_lens, class_ids, keys, memory = prepare_data(data)
        mem_embedding = netT(memory)
        mem_embedding = torch.squeeze(mem_embedding).contiguous()
        hidden = rnn_model.init_hidden(batch_size)
        _, sent_emb = rnn_model(captions, cap_lens, hidden, mem_embedding)

        Inception_inp = imgs[0]*2 - 1
        _, sent_code = inception_model(Inception_inp)
        imgs_cos = []
        for i in range(100):
            image_code = sent_code[i]
            image_code = image_code.unsqueeze(0)
            image_code = image_code.repeat(100, 1)
            img_cos = cosine_similarity(image_code, sent_emb)
            img_cos = img_cos
            _, indices = torch.sort(img_cos, descending=True)
            top = indices[:1]
            if i in top:
                imgs_cos.append(1)
            else:
                imgs_cos.append(0)
        P_rate = sum(imgs_cos)
        P_rates.append(P_rate)
        print('step: %d | precision: %d' % (step, P_rate))

    start = valid_dir.find('netG')
    end = valid_dir.find('valid')
    epoch = valid_dir[start:end-1]
    A_precision = sum(P_rates) * 1.0 / len(P_rates)
    print('%s average R_precsion is %f | total time: %f' % (epoch, A_precision, time.time() - start_time))

