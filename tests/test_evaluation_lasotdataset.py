import os
import sys
# 确保项目路径在 sys.path 中（如果直接运行）
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lib.test.evaluation.lasotdataset import LaSOTDataset


def test_lasot_basic():
    """基础测试：实例化、检查序列数量、首序列路径、ground_truth 格式"""
    dataset = LaSOTDataset()   # 如果想测训练集，改为 'train'

    # 2. 检查序列列表长度（应为 280）
    print(f"总序列数: {len(dataset.sequence_list)}")
    assert len(dataset.sequence_list) == 280, f"Expected 280, got {len(dataset.sequence_list)}"

    # 3. 检查 clean_list（类别列表）长度
    print(f"类别数: {len(set(dataset.clean_list))}")
    assert len(set(dataset.clean_list)) == 70, "Expected 70 classes"

    # 4. 获取第一个序列
    seq_list = dataset.get_sequence_list()
    first_seq = seq_list[0]
    print(f"\n第一个序列名称: {first_seq.name}")
    print(f"图片数量: {len(first_seq.frames)}")
    print(f"Ground truth 形状: {first_seq.ground_truth_rect.shape}")
    print(f"目标可见性长度: {len(first_seq.target_visible)}")
    print(f"第一帧 bbox: {first_seq.ground_truth_rect[0]}")

    # 5. 检查文件是否存在（路径有效性）
    first_frame_path = first_seq.frames[0]
    print(f"第一帧图片路径: {first_frame_path}")
    assert os.path.isfile(first_frame_path), f"图片不存在: {first_frame_path}"

    # 6. 检查标注文件是否存在
    # 根据 _construct_sequence 构建 anno_path，这里我们手动构造
    class_name = first_seq.object_class
    seq_name = first_seq.name
    anno_path = os.path.join(dataset.base_path, class_name, seq_name, "groundtruth.txt")
    print(f"标注文件路径: {anno_path}")
    assert os.path.isfile(anno_path), f"标注文件不存在: {anno_path}"

    print("\n✅ 所有基础测试通过！")


if __name__ == "__main__":
    test_lasot_basic()