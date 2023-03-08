# ------------------------------------------------------------------------
# Modified from TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2012. Xiaolong Liu
# ------------------------------------------------------------------------


import datetime
import json
import random
import time
from pathlib import Path
import re
import os
import logging
import sys
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from video_encoder import VideoEncoder

from datasets import build_dataset
from engine import train_one_epoch, test
from models import build_model
if cfg.tensorboard:
    from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'




def main(args):
    
    model = VideoEncoder(args.encoder)



    model.to(device)


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    elif args.multi_gpu:
        model = torch.nn.DataParallel(model)


    dataset = build_dataset(subset=cfg.test_set, args=cfg, mode='val')

    if args.distributed:
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    data_loader_val = DataLoader(dataset_val, cfg.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    s_ = time.time()
    main(args)
    logging.info('main takes {:.3f} seconds'.format(time.time() - s_))
