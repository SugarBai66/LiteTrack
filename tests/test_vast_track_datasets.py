import pytest
import torch
from lib.train.dataset.vast_track import VastTrack

def test_get_sequence_info():
    dataset = VastTrack()
    # dataset = VastTrack(root='/mnt/ssd4t/datasets/VastTrack/train')
    seq_id = 0
    info = dataset.get_sequence_info(seq_id)
    assert 'bbox' in info
    assert info['bbox'].shape[1] == 4
    assert info['bbox'].dtype == torch.float32
    assert info['valid'].dtype == torch.bool or torch.uint8

def test_get_frames():
    dataset =  VastTrack(root='/path/to/root')
    seq_id = 0
    frame_ids = [0, 10, 20]
    frames, anno_frames, meta = dataset.get_frames(seq_id, frame_ids)
    assert len(frames) == len(frame_ids)
    assert len(anno_frames['bbox']) == len(frame_ids)
    assert frames[0].shape[2] == 3  # 检查图片通道数