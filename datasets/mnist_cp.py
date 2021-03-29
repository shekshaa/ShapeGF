import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import tqdm
import glob
from datasets.utils import read_ply


class MNIST_CP(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size, te_sample_size,
        split='train', scale=1., normalize_per_shape=False, random_subsample=False,
        normalize_std_per_axis=False, recenter_per_shape=False, all_points_mean=None,
        all_points_std=None, input_dim=3):
        
        super(MNIST_CP, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        
        # subdirs can be from 0 to 9
        for cate_idx, subd in tqdm.tqdm(enumerate(self.subdirs), total=len(self.subdirs)):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            # for x in os.listdir(sub_path):
            #     if not x.endswith('.ply'):
            #         continue
            #     all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>"
            # or "val/<mid>" or "test/<mid>"

            for mid in glob.glob(sub_path + '/*.ply'):
                # obj_fname = os.path.join(sub_path, x)
                # obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                try:
                    point_cloud = read_ply(mid)
                except:  # nofa: E722
                    continue

                assert point_cloud.shape[0] == 800
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 800, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.recenter_per_shape = recenter_per_shape

        if all_points_mean is not None and \
            all_points_std is not None and \
                not self.recenter_per_shape:
            # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.recenter_per_shape:  # per shape center
            # TODO: bounding box scale at the large dim and center
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)
            ) / 2
            self.all_points_std = np.amax((
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)
            ), axis=-1).reshape(B, 1, 1) / 2
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(
                axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(
                -1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    -1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        self.train_points = self.all_points[:, :600]  # TODO: hard-coded index
        self.test_points = self.all_points[:, 600:]

        self.tr_sample_size = min(600, tr_sample_size)
        self.te_sample_size = min(200, te_sample_size)

        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def get_pc_stats(self, idx):
        if self.recenter_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), \
            self.all_points_std.reshape(1, -1)

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'tr_points': tr_out,
            'te_points': te_out,
            'mean': m, 'std': s, 'cate_idx': cate_idx,
            'sid': sid, 'mid': mid,
            'display_axis_order': self.display_axis_order
        }
