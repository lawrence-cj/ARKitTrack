import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import cv2
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class ARKit(BaseVideoDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().got10k_dir if root is None else root
        super().__init__('iRGBD', root, image_loader)


        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if split == 'train':
                self.root = os.path.join(self.root, 'vot_train')
                self.sequence_list = sorted(os.listdir(self.root))
                self.sequence_list = [s for s in self.sequence_list if '.' not in s]
            elif split == 'test':
                self.root = os.path.join(self.root, 'vot_test')
                self.sequence_list = sorted(os.listdir(self.root))
                self.sequence_list = [s for s in self.sequence_list if '.' not in s]
            else:
                raise ValueError('Unknown split name.')
        else:
            raise ValueError('Unknown split name.')

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        # self.sequence_meta_info = self._load_meta_info()
        # self.seq_per_class = self._build_seq_per_class()

        # self.class_list = list(self.seq_per_class.keys())
        # self.class_list.sort()

    def get_name(self):
        return 'irgbd'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, na_filter=False, low_memory=False).values
        gt = gt.astype(np.float32)
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id]),  os.path.join(self.root, self.sequence_list[seq_id], 'depth')

    def get_sequence_info(self, seq_id):
        seq_path, seqD_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 10) & (bbox[:, 3] > 10)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frameD_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.png'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_frameD(self, seq_path, frame_id):
        depth = cv2.imread(self._get_frameD_path(seq_path, frame_id), cv2.IMREAD_UNCHANGED) / 10  # mm -> cm

        max_depth = min(np.median(depth) * 3, 1000)  # 1000cm
        depth[depth > max_depth] = max_depth

        # depth = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        depth = cv2.resize(depth, (1920, 1440))

        return depth

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None, need_depth=False):
        seq_path, seqD_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if need_depth:
            frameD_list = [self._get_frameD(seqD_path, f_id) for f_id in frame_ids]
        else:
            frameD_list = None

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        if need_depth:
            return frame_list, frameD_list, anno_frames, obj_meta
        else:
            return frame_list, anno_frames, obj_meta

