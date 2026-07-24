import pytest
import torch
from lib.train.dataset.vast_track import VastTrack

def test_get_sequence_info():
    dataset = VastTrack()  # 或显式传 root
    seq_id = -1
    seq_path = dataset._get_sequence_path(seq_id)
    print(f"seq_path: {seq_path}")

    # 检查目录是否存在
    import os
    print(f"Directory exists: {os.path.isdir(seq_path)}")

    # 列出目录下的所有文件，看看是否有 Groundtruth.txt
    print(f"Files in {seq_path}: {os.listdir(seq_path)}")

    anno_file = os.path.join(seq_path, "Groundtruth.txt")
    print(f"anno_file: {anno_file}")
    print(f"File exists: {os.path.isfile(anno_file)}")

    # 如果文件存在，尝试直接读取前几行
    if os.path.isfile(anno_file):
        with open(anno_file, 'r') as f:
            lines = f.readlines()
            print(f"First 2 lines: {lines[:3]}")

    # 然后再调用 get_sequence_info
    info = dataset.get_sequence_info(seq_id)

def test_get_frames():
    dataset =  VastTrack()
    seq_id = 0
    frame_ids = [0, 10, 20]
    frames, anno_frames, meta = dataset.get_frames(seq_id, frame_ids)
    assert len(frames) == len(frame_ids)
    assert len(anno_frames['bbox']) == len(frame_ids)
    assert frames[0].shape[2] == 3  # 检查图片通道数