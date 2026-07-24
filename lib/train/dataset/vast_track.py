import os
import torch
import numpy as np
import pandas
from collections import OrderedDict
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from .base_video_dataset import BaseVideoDataset

class VastTrack(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe,
                 split='train', seq_ids=None, data_fraction=None):
        # 如果 root 为空，从环境配置中读取
        root = env_settings().vasttrack_dir if root is None else root
        super().__init__('VastTrack', root, image_loader)

        # 1. 构建序列列表（可以是目录遍历，也可以从 list.txt 读取）
        self.sequence_list = []
        # 假设所有类别文件夹在第一层
        for cls in os.listdir(root):
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue
            for vid in os.listdir(cls_path):
                vid_path = os.path.join(cls_path, vid)
                if os.path.isdir(vid_path):
                    self.sequence_list.append((cls, vid))  # 保存类别和视频名

        # 2. 如果指定了 seq_ids，筛选子集
        if seq_ids is not None:
            self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        # 3. 可选数据采样
        if data_fraction is not None:
            import random
            self.sequence_list = random.sample(self.sequence_list,
                                               int(len(self.sequence_list)*data_fraction))

        # 4. 构建类别索引（用于按类采样）
        self.seq_per_class = self._build_class_list()

    def _build_class_list(self):
        seq_per_class = {}
        for idx, (cls, vid) in enumerate(self.sequence_list):
            seq_per_class.setdefault(cls, []).append(idx)
        return seq_per_class

    def get_name(self):
        return 'vast_track'

    def has_class_info(self):
        return True  # 如果有类别信息

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.seq_per_class)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class.get(class_name, [])

    def _get_sequence_path(self, seq_id):
        cls, vid = self.sequence_list[seq_id]
        return os.path.join(self.root, cls, vid)

    def _read_bb_anno(self, seq_path):
        # 根据实际标注文件格式修改
        anno_file = os.path.join(seq_path, "groundtruth.txt")
        # 如果分隔符是逗号
        gt = pandas.read_csv(anno_file, delimiter=',', header=None,
                             dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt, dtype=torch.float32)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # 如果没有单独的可见性标签，将 visible 设为 valid
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # 修改图片命名格式（例如帧号从 1 开始，补零）
        return os.path.join(seq_path, 'imgs', '{:05d}.jpg'.format(frame_id+1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        return self.sequence_list[seq_id][0]  # 返回类别名

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': self.get_class_name(seq_id),
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list, anno_frames, object_meta