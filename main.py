# ------------------------------------------------------------------------
# Modified from TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2012. Xiaolong Liu
# ------------------------------------------------------------------------


import datetime
import json
import random
import time
from tqdm import tqdm
import re
import os
import sys
from ruamel import yaml
import numpy as np
import torch
from video_encoder import VideoEncoder
from datasets.dataset import collate_fn
from datasets.misc import setup_for_distributed, init_distributed_mode
import logging
from datasets import build_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'


def get_configs(dataset):
    default_config = yaml.load(
        open('datasets/dataset_cfg.yaml', 'r'), Loader=yaml.RoundTripLoader)[dataset]
    return default_config


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = get_configs(args.dataset)
    args.__dict__.update(configs)
    init_distributed_mode(args)
    model = VideoEncoder(args.backbone, args)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    dataset = build_dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)

    model.train(False)
    features = dict()
    for data in tqdm(dataloader):
        # get the inputs

        vids, input_data, targets, num_frames, bases = data
        input_data.tensors = input_data.tensors.to(device)
        input_data.mask = input_data.mask.to(device)
        output = model(input_data)
        feature = output.tensors.cpu().detach().numpy()
        mask = output.mask.cpu().detach().numpy()
        for i, vid in enumerate(vids):
            if vid not in features:
                features[vid] = {}
                features[vid]['feature'] = []
                features[vid]['num_frames'] = num_frames[i]
            diction = {}
            diction['embedding'], diction['mask'], diction['target'], diction['base'] = feature[i], mask[i], targets[i], bases[i]
            features[vid]['feature'].append(diction)

    # create dir
    new_path = os.path.join(args.save_dir, args.backbone, args.dataset)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # save features
    np.save(os.path.join(new_path, f'{args.split}_data.npy'), features)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Video Feature Extractor', add_help=False)
    parser.add_argument('--dataset', default='thumos14', type=str)
    parser.add_argument('--backbone', default='i3d', type=str)
    parser.add_argument('--split', default='training', type=str)
    parser.add_argument('--save_dir', default='output_features', type=str)
    parser.add_argument('--img_tensor', default=False, type=bool)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    s_ = time.time()
    main(args)
    logging.info('main takes {:.3f} seconds'.format(time.time() - s_))
