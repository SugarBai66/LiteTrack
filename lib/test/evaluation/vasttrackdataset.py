import numpy as np
import os
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class VastTrackDataset(BaseDataset):
    def __init__(self, split='test'):
        """
        VastTrack test set
        split: 'test' (or 'test_all' depending on your folder name)
        """
        super().__init__()
        # 从 local.py 中读取 vast_track_path
        self.base_path = self.env_settings.vast_track_path
        # 拼接测试集子目录（如果你的测试集叫 test_all，就改成 test_all）
        self.test_path = os.path.join(self.base_path, split)  # 例如 /mnt/ssd4t/datasets/VastTrack/test
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, seq_name):
        # seq_name: (class_name, video_name) 元组，例如 ('Aardwolf', 'Aardwolf-10')
        class_name, video_name = seq_name
        seq_path = os.path.join(self.test_path, class_name, video_name)
        anno_path = os.path.join(seq_path, 'Groundtruth.txt')
        # 加载标注，逗号分隔
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        # 图片路径列表
        num_frames = ground_truth_rect.shape[0]
        img_dir = os.path.join(seq_path, 'imgs')
        frames_list = [os.path.join(img_dir, f'{i:05d}.jpg') for i in range(1, num_frames+1)]
        # 目标可见性：VastTrack 没有提供单独的可见性文件，默认全部可见
        target_visible = np.ones(num_frames, dtype=bool)
        # 返回 Sequence 对象
        return Sequence(
            name=f"{class_name}/{video_name}",
            frames=frames_list,
            dataset='vasttrack',
            ground_truth_rect=ground_truth_rect.reshape(-1, 4),
            object_class=class_name,
            target_visible=target_visible
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        """遍历测试集目录，收集所有 (类别, 视频) 元组"""
        seq_list = []
        for cls in os.listdir(self.test_path):
            cls_path = os.path.join(self.test_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for vid in os.listdir(cls_path):
                vid_path = os.path.join(cls_path, vid)
                if os.path.isdir(vid_path):
                    # 检查 Groundtruth.txt 是否存在，防止空文件夹
                    if os.path.exists(os.path.join(vid_path, 'Groundtruth.txt')):
                        seq_list.append((cls, vid))
        return seq_list