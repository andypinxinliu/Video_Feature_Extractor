'''Utilities for data loading'''

import json
import math
import logging
import os

import pandas as pd
import easydict
import yaml

import numpy as np
import torch
import torch.nn.functional as F

import os

import numpy as np
import cv2


def load_json(path):
    return json.load(open(path))

def load_video_frames(frame_dir, start, seq_len, stride=1, fn_tmpl='img_%07d.jpg'):
    '''
    Load a sequence of video frames into memory. 

    Params:
        frame_dir: the directory to the decoded video frames
        start: load image starting with this index. 1-indexed
        seq_len: length of the loaded sub-sequence.
        stride: load one frame every `stride` frame from the sequence.
    Returns:
        Nd-array with shape (T, H, W, C) in float32 precision. T = num // stride
    '''
    frames = []
    if seq_len > 0:
        # load a fixed-length frame sequence
        # for i in range(start + stride // 2, start + seq_len, stride):
        #     img = cv2.imread(os.path.join(frame_dir, fn_tmpl % i))
        #     if  img is None:
        #         # print('failed to load {}'.format(os.path.join(frame_dir, fn_tmpl % i)))
        #         raise IOError(os.path.join(frame_dir, fn_tmpl % i))
        #     # img = img[:, :, [2, 1, 0]]  # BGR => RGB, moved to video_transforms.Normalize
        #     # img = (img/255.)*2 - 1
        #     frames.append(img)
        frames = [cv2.imread(os.path.join(frame_dir, fn_tmpl % i))
            for i in range(start + stride // 2, start + seq_len, stride)]
    else:
        # load all frames
        num_imgs = len(os.listdir(frame_dir))
        frames = [cv2.imread(os.path.join(frame_dir, fn_tmpl % (i+1))) for i in range(num_imgs)]
    return np.asarray(frames, dtype=np.float32)  # NHWC


def make_img_transform(is_training, resize=110, crop=96, mean=127.5, std=127.5, keep_asr=True):
    from .videotransforms import GroupResizeShorterSide, GroupCenterCrop, GroupRandomCrop, GroupRandomHorizontalFlip, GroupPhotoMetricDistortion, GroupRotate, GroupResize, GroupNormalize
    from torchvision.transforms import Compose

    if isinstance(resize, (list, tuple)):
        resize_trans = GroupResize(resize)
    else:
        if keep_asr:
            assert isinstance(resize, int), 'if keep asr, resize must be a single integer'
            resize_trans = GroupResizeShorterSide(resize)
        else:
            resize_trans = GroupResize((resize, resize))

    transforms = [
        resize_trans,
        GroupRandomCrop(crop) if is_training else GroupCenterCrop(crop),
    ]
    if is_training:
            transforms += [
                GroupPhotoMetricDistortion(brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18,
                    p=0.5),
                GroupRotate(limit=(-45, 45),
                    border_mode='reflect101',
                    p=0.5),
                GroupRandomHorizontalFlip(0.5),
            ]
        
    transforms.append(GroupNormalize(mean, std, to_rgb=True))
    return Compose(transforms)


def get_valid_anno(gt_instances, slice, thr=0.75,
        start_getter=lambda x: x['segment'][0],
        end_getter=lambda x: x['segment'][1]):
    '''Perform integrity based instance filtering'''
    start, end = slice
    kept_instances = []
    for inst in gt_instances:
        # ignore insts outside the time window (slice)
        if end_getter(inst) <= start or start_getter(inst) >= end:
            continue
        else:
            # clamped inst
            new_start = max(start_getter(inst), start)
            new_end = min(end_getter(inst), end)
            integrity = (new_end - new_start) * 1.0 / (end_getter(inst) - start_getter(inst))
            
            if integrity >= thr:
                new_inst = {k:v for k,v in inst.items()}
                new_inst['segment'] = [new_start - start, new_end - start]
                kept_instances.append(new_inst)
    return kept_instances


def get_dataset_dict(video_info_path, video_anno_path, subset, mode='test', exclude_videos=None, online_slice=False, slice_len=None, ignore_empty=True, slice_overlap=0, return_id_list=False):
    '''
    Prepare a dict that contains the information of each video, such as duration, annotations.
    Args:
        video_info_path: path to the video info file in json format. This file records the length and fps of each video.
        video_anno_path: path to the ActivityNet-style video annotation in json format.
        subset: e.g. train, val, test
        mode: train (for training) or test (for inference).
        online_slice: cut videos into slices for training and testing. It should be enabled if the videos are too long.
        slice_len: length of video slices.
        ignore_empty: ignore video slices that does not contain any action instance. This should be enabled only in the training phase.
        slice_overlap: overlap ration between adjacent slices (= overlap_length / slice_len)

    Return:
        dict
    '''
    video_ft_info = load_json(video_info_path)
    anno_data = load_json(video_anno_path)['database']

    video_dict = {}
    id_list = []
    cnt = 0

    video_set = set([x for x in anno_data if anno_data[x]['subset'] in subset])
    video_set = video_set.intersection(video_ft_info.keys())

    if exclude_videos is not None:
        assert isinstance(exclude_videos, (list, tuple))
        video_set = video_set.difference(exclude_videos)

    video_list = list(sorted(video_set))

    for video_name in video_list:
        # remove ambiguous instances on THUMOS14
        annotations = [x for x in anno_data[video_name]['annotations'] if x['label'] != 'Ambiguous']
        annotations = list(sorted(annotations, key=lambda x: sum(x['segment'])))

        if video_name in video_ft_info:
            # video_info records the length in snippets, duration and fps (#frames per second) of the feature/image sequence
            video_info = video_ft_info[video_name]
            # number of frames or snippets
            feature_length = int(video_info['feature_length'])   
            feature_fps = video_info['feature_fps']
            feature_second = video_info['feature_second']
        else:
            continue

        video_subset = anno_data[video_name]['subset']
        # For THUMOS14, we crop video into slices of fixed length
        if online_slice:
            stride = slice_len * (1 - slice_overlap)

            if feature_length <= slice_len:
                slices = [[0, feature_length]]
            else:
                # stride * (i - 1) + slice_len <= feature_length
                # i <= (feature_length - slice_len)
                num_complete_slices = int(math.floor(
                    (feature_length / slice_len - 1) / (1 - slice_overlap) + 1))
                slices = [
                    [int(i * stride), int(i * stride) + slice_len] for i in range(num_complete_slices)]
                if (num_complete_slices - 1) * stride + slice_len < feature_length:
                    # if video_name == 'video_test_0000006':
                    #     pdb.set_trace()
                    if mode != 'train':
                        # take the last incomplete slice
                        last_slice_start = int(stride * num_complete_slices)
                    else:
                        # move left to get a complete slice.
                        # This is a historical issue. The performance might be better
                        # if we keep the same rule for training and inference 
                        last_slice_start = max(0, feature_length - slice_len)
                    slices.append([last_slice_start, feature_length])
            num_kept_slice = 0
            for slice in slices:
                time_slices = [slice[0] / video_info['feature_fps'], slice[1] / video_info['feature_fps']]
                feature_second = time_slices[1] - time_slices[0]
                # perform integrity-based instance filtering
                valid_annotations = get_valid_anno(annotations, time_slices)
                
                if not ignore_empty or len(valid_annotations) >= 1:
                    # rename the video slice
                    new_vid_name = video_name + '_window_{}_{}'.format(*slice)
                    new_vid_info = {
                        'annotations': valid_annotations, 'src_vid_name': video_name, 
                        'feature_fps': feature_fps, 'feature_length': slice_len, 
                        'subset': subset, 'feature_second': feature_second, 'time_offset': time_slices[0]}
                    video_dict[new_vid_name] = new_vid_info
                    id_list.append(new_vid_name)
                    num_kept_slice += 1
            if num_kept_slice > 0:
                cnt += 1
        # for ActivityNet and hacs, use the full-length videos as samples
        else:
            if not ignore_empty or len(annotations) >= 1:
                # Remove incorrect annotions on ActivityNet
                valid_annotations = [x for x in annotations if x['segment'][1] - x['segment'][0] > 0.02]

                if ignore_empty and len(valid_annotations) == 0:
                    continue
                
                video_dict[video_name] = {
                    'src_vid_name': video_name, 'annotations': valid_annotations, 
                    'feature_fps': feature_fps, 'feature_length': int(feature_length),
                    'subset': video_subset, 'feature_second': feature_second, 'time_offset': 0}
                id_list.append(video_name)
                cnt += 1
    logging.info('{} videos, {} slices'.format(cnt, len(video_dict)))
    if return_id_list:
        return video_dict, id_list
    else:
        return video_dict