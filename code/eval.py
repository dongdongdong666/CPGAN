from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from miscc.utils import mkdir_p
from datasets import prepare_data
from PIL import Image
from miscc.utils import truncated_normal_

import os
import sys
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import h5py
import torch.backends.cudnn as cudnn
from nltk.tokenize import RegexpTokenizer

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    # args = parser.parse_args('--cfg cfg/eval_coco.yml --gpu 0'.split())
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    B_VALIDATION = cfg.B_VALIDATION
    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    args.manualSeed = 100
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir = 'test'
    bshuffle = True

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    netT = torch.load(cfg.TRAIN.NET_T)
    text_encoder = torch.load(cfg.TRAIN.NET_E)
    netG = torch.load(cfg.TRAIN.NET_G)

    netT = netT.cuda()
    netG = netG.cuda()
    text_encoder = text_encoder.cuda()

    if split_dir == 'test':
        split_dir = 'valid'

    batch_size = cfg.TRAIN.BATCH_SIZE
    nz = cfg.GAN.Z_DIM
    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
    noise = noise.cuda()
    save_dir = '../outputs/Inference_Images'
    mkdir_p(save_dir)

    if B_VALIDATION:
        cnt = 0
        cap_dir = save_dir + '/caption.h5'
        cap_len_dir = save_dir + '/caption_len.h5'
        memory_dir = save_dir + '/memory.h5'
        f_cap = h5py.File(cap_dir, 'w')
        f_len = h5py.File(cap_len_dir, 'w')
        f_memory = h5py.File(memory_dir, 'w')
        for step, data in enumerate(dataloader, 0):
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)

            # imgs, captions, cap_lens, class_ids, keys, memory = prepare_data(data)
            captions, cap_lens, class_ids, keys, memory = prepare_data(data)
            for i in range(batch_size):
                key = keys[i]
                caption = captions[i]
                caption = caption.unsqueeze(1)
                caption = caption.cpu()
                caption = caption.numpy()
                f_cap[key] = caption
                mem = memory[i]
                mem = torch.transpose(mem, 1, 0)
                mem = mem.cpu()
                mem = mem.numpy()
                f_memory[key] = mem
                cap_len = cap_lens[i].item()
                f_len[key] = cap_len

            memory = torch.transpose(memory, 1, 2).contiguous()
            mem_embedding = netT(memory)
            mem_embedding = torch.squeeze(mem_embedding).contiguous()
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden, mem_embedding)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            #  get mask
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            truncated_normal_(noise, 0, 1)
            fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
            for j in range(batch_size):
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                k = -1
                # for k in range(len(fake_imgs)):
                im = fake_imgs[k][j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_s%d.png' % (s_tmp, k)
                im.save(fullpath)
        f_cap.close()
        f_len.close()
        f_memory.close()
    else:

        wordtoix = dataset.wordtoix
        filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
        data_dic = {}
        with open(filepath, "r") as f:
            filenames = f.read().decode('utf8').split('\n')
            for name in filenames:
                if len(name) == 0:
                    continue
                filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
                with open(filepath, "r") as f:
                    print('Load from:', name)
                    sentences = f.read().decode('utf8').split('\n')
                    # a list of indices for a sentence
                    captions = []
                    cap_lens = []
                    for sent in sentences:
                        if len(sent) == 0:
                            continue
                        sent = sent.replace("\ufffd\ufffd", " ")
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(sent.lower())
                        if len(tokens) == 0:
                            print('sent', sent)
                            continue

                        rev = []
                        for t in tokens:
                            t = t.encode('ascii', 'ignore').decode('ascii')
                            if len(t) > 0 and t in wordtoix:
                                rev.append(wordtoix[t])
                        captions.append(rev)
                        cap_lens.append(len(rev))
                max_len = np.max(cap_lens)

                sorted_indices = np.argsort(cap_lens)[::-1]
                cap_lens = np.asarray(cap_lens)
                cap_lens = cap_lens[sorted_indices]
                cap_array = np.zeros((len(captions), max_len), dtype='int64')
                for i in range(len(captions)):
                    idx = sorted_indices[i]
                    cap = captions[idx]
                    c_len = len(cap)
                    cap_array[i, :c_len] = cap
                key = name[(name.rfind('/') + 1):]
                data_dic[key] = [cap_array, cap_lens, sorted_indices]

        s_tmp = save_dir
        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            mkdir_p(save_dir)
            captions, cap_lens, sorted_indices = data_dic[key]
            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM
            captions = Variable(torch.from_numpy(captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()
            memory_dir = '../memory/memoryft_9487.pkl'
            vocab_trans_dir = '../memory/transvob.pkl'
            f_memory = open(memory_dir, 'rb')
            memory_feats = pickle.load(f_memory)
            f_memory.close()
            f_trans = open(vocab_trans_dir, 'rb')
            vob_trans = pickle.load(f_trans)
            f_trans.close()
            memory_list = []
            for i in range(captions.shape[0]):
                word_list = []
                for j in range(captions[i].shape[0]):
                    cap_id = vob_trans[captions[i][j]]
                    word_list.append(torch.from_numpy(memory_feats[str(cap_id)]).unsqueeze(dim=0))
                sentence = torch.cat(word_list, 0)
                memory_list.append(sentence.unsqueeze(dim=0))
            memory = torch.cat(memory_list, 0)
            memory = memory.cuda()
            memory = torch.transpose(memory, 1, 2).contiguous()
            mem_embs = netT(memory)
            mem_embs = torch.squeeze(mem_embs).contiguous()
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden, mem_embs)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            truncated_normal_(noise, 0, 1)
            fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
            for j in range(batch_size):
                fake_img = fake_imgs[2][j]
                fake_im = Image.fromarray(np.transpose(((fake_img.detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8), (1,2,0)))
                fullpath = '%s/%d.png' % (save_dir,j)
                fake_im.save(fullpath)







