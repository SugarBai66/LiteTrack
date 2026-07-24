import os
import sys

# sys.path.append('.')  # 确保项目根目录在 Python 路径中

from lib.test.evaluation.vasttrackdataset import VastTrackDataset


def test_dataset_loading():
    # 1. 实例化数据集（使用 split='test'，根据实际情况改为 'test_all'）
    dataset = VastTrackDataset(split='test')

    # 2. 打印数据集大小
    print(f"Total sequences: {len(dataset)}")

    # 3. 获取序列列表
    seq_list = dataset.get_sequence_list()
    print(f"Sequence list length: {len(seq_list)}")

    # 4. 查看前几个序列的名称
    for i in range(min(5, len(seq_list))):
        seq = seq_list[i]
        print(f"Sequence {i}: {seq.name}, frames: {len(seq.frames)}, gt shape: {seq.ground_truth_rect.shape}")

    # 5. 选取第一个序列，详细检查
    if len(seq_list) > 0:
        seq0 = seq_list[0]
        print("\nDetailed check of first sequence:")
        print(f"Name: {seq0.name}")
        print(f"Object class: {seq0.object_class}")
        print(f"Number of frames: {len(seq0.frames)}")
        print(f"Ground truth shape: {seq0.ground_truth_rect.shape}")  # 应为 (N,4)
        print(f"Target visible length: {len(seq0.target_visible)}")

        # 检查前几个帧文件是否存在
        for i, fpath in enumerate(seq0.frames[:5]):
            exists = os.path.exists(fpath)
            print(f"Frame {i}: {os.path.basename(fpath)} exists? {exists}")

        # 检查标注数据格式
        print(f"First 3 ground truth boxes:\n{seq0.ground_truth_rect[:3]}")

    # 6. 随机检查若干序列的图片完整性（可选，比较耗时）
    # import random
    # sample_indices = random.sample(range(len(seq_list)), min(10, len(seq_list)))
    # for idx in sample_indices:
    #     seq = seq_list[idx]
    #     missing = [f for f in seq.frames if not os.path.exists(f)]
    #     if missing:
    #         print(f"WARNING: {seq.name} has {len(missing)} missing images")



if __name__ == '__main__':
    test_dataset_loading()