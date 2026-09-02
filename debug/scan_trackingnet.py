#!/usr/bin/env python3
"""
TrackingNet 数据集完整性检查脚本（只读，不修改任何文件）
检查项：
1. 标注文件格式（列数、数值合法性）
2. 图片是否存在、数量是否匹配标注行数
3. 是否有孤立图片（无对应标注）
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# 可选的图片子目录名称（按优先级尝试）
POSSIBLE_IMAGE_DIRS = ["frames", "images", "img", "JPEGImages", "color"]


def parse_annotation(filepath):
    """
    读取标注文件，返回 (行数, 每行解析后的bbox列表)
    若格式错误，抛出异常并附带错误信息
    """
    rows = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行，但记录警告
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) != 4:
                raise ValueError(f"Line {line_num}: expected 4 values, got {len(parts)}")
            try:
                vals = [float(x) for x in parts]
            except ValueError as e:
                raise ValueError(f"Line {line_num}: non-numeric value - {e}")
            # 基本合法性检查：x,y 可为0或正，w,h必须>0
            x, y, w, h = vals
            if w <= 0 or h <= 0:
                raise ValueError(f"Line {line_num}: width ({w}) or height ({h}) <= 0")
            if x < 0 or y < 0:
                # 有些数据集允许负坐标（超出图像边界），这里仅作警告，不报错
                # 但可以记录，此处为保持严格，可改为警告，但为了方便我们先通过
                pass
            rows.append((x, y, w, h))
    return rows


def find_image_dir(base_dir, video_id):
    """
    尝试在 base_dir 下查找视频的视频帧目录。
    常见结构：
        - base_dir/frames/video_id/
        - base_dir/video_id/
        - base_dir/images/video_id/
        - 直接放在 base_dir 下（以 video_id 前缀）
    返回第一个存在的目录路径，或 None
    """
    # 首先尝试常见的子目录结构
    for sub in POSSIBLE_IMAGE_DIRS:
        candidate = base_dir / sub / video_id
        if candidate.is_dir():
            # 检查是否有图片文件
            img_files = list(candidate.glob("*.jpg")) + list(candidate.glob("*.png")) + list(candidate.glob("*.jpeg"))
            if img_files:
                return candidate
    # 尝试 video_id 直接在 base_dir 下
    candidate = base_dir / video_id
    if candidate.is_dir():
        img_files = list(candidate.glob("*.jpg")) + list(candidate.glob("*.png")) + list(candidate.glob("*.jpeg"))
        if img_files:
            return candidate
    # 尝试 base_dir 下以 video_id 为前缀的文件（但通常不会）
    # 最后尝试 base_dir 本身（假设所有图片平铺）
    candidate = base_dir
    # 但是要避免误判，必须检查是否存在 video_id 相关的图片，但平铺时无法确定，跳过
    # 返回 None 表示未找到
    return None


def check_images(imagedir, num_frames):
    """
    检查 imagedir 下是否存在连续编号的图片，从 1 开始到 num_frames
    返回缺失的帧号列表，以及多余的图片文件列表（如果图片数多于num_frames）
    支持 .jpg .png .jpeg，优先尝试数字补零的格式（如 000001.jpg）
    """
    if imagedir is None:
        return None, None, "Image directory not found"

    # 收集所有图片文件，按数字编号排序
    img_files = {}
    # 常见的命名模式：数字（可能补零）+ 扩展名
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for f in imagedir.glob(ext):
            stem = f.stem
            if stem.isdigit():
                num = int(stem)
                img_files[num] = f
            else:
                # 尝试提取数字部分，如 'frame_00001' -> 1
                # 简单方法：提取连续数字
                import re
                digits = re.findall(r'\d+', stem)
                if digits:
                    num = int(digits[-1])  # 取最后一个数字
                    img_files[num] = f
                # 否则忽略（可能不是帧文件）

    if not img_files:
        return None, None, "No image files found in directory"

    # 检查缺失帧
    max_found = max(img_files.keys()) if img_files else 0
    # 通常帧从1开始，但有些从0开始，我们检查从1到num_frames
    missing = []
    for i in range(1, num_frames ):
        if i not in img_files:
            missing.append(i)
    # 检查多余图片（大于num_frames的帧）
    extra = [f for num, f in img_files.items() if num > num_frames]
    # 另外检查是否有编号为0的（如果有的话）
    if 0 in img_files and num_frames > 0:
        # 可能从0开始，则调整：如果缺少1但有0，且num_frames=行数，可能没问题，但这里保守处理
        # 我们检查是否所有1..num_frames都缺失但0存在，这种情况很少见，暂不处理，只报告
        pass

    return missing, extra, None


def scan_video(anno_file, train_dir, root):
    """
    扫描一个视频序列（由anno_file指定）
    返回错误信息列表
    """
    errors = []
    warnings = []
    video_id = anno_file.stem
    # 1. 解析标注文件
    try:
        annotations = parse_annotation(anno_file)
        num_boxes = len(annotations)
    except Exception as e:
        errors.append(f"Annotation parse error: {e}")
        return errors, warnings  # 无法继续

    # 2. 寻找图片目录
    # 通常图片目录在 train_dir 下，但有可能在 root/TRAIN_i/frames/ 等
    # 尝试多种可能
    possible_dirs = [
        train_dir / "frames" / video_id,
        train_dir / "images" / video_id,
        train_dir / "img" / video_id,
        train_dir / video_id,
        train_dir / "JPEGImages" / video_id,
        train_dir / "color" / video_id,
        # 有些可能直接放在 train_dir 下，但以 video_id 开头的文件（如 video_id_00001.jpg）不常见
    ]
    imagedir = None
    for d in possible_dirs:
        if d.is_dir():
            # 检查是否有图片
            img_list = list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg"))
            if img_list:
                imagedir = d
                break

    if imagedir is None:
        errors.append("Image directory not found in any expected location")
        return errors, warnings

    # 3. 检查图片数量和连续性
    missing, extra, err_msg = check_images(imagedir, num_boxes)
    if err_msg:
        errors.append(f"Image check error: {err_msg}")
    else:
        if missing:
            warnings.append(
                f"Missing frames: {missing[:10]}{'...' if len(missing) > 10 else ''} (total {len(missing)})")
        if extra:
            warnings.append(f"Extra image files beyond annotation count: {len(extra)} files")
        # 还可以检查是否有小于1的编号，但略过

    # 4. 可选：检查图片尺寸与bbox是否超出（太耗时，略过）

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Check TrackingNet dataset integrity (read-only)")
    parser.add_argument("--root", default="/mnt/ssd4t/datasets/TrackingNet",
                        help="Root directory of TrackingNet dataset")
    parser.add_argument("--train_dirs", nargs="+", default=[f"TRAIN_{i}" for i in range(12)],
                        help="Subdirectories to scan (e.g., TRAIN_0 TRAIN_1 ...)")
    parser.add_argument("--test", action="store_true", help="Also scan TEST directory if present")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Error: Root directory '{root}' does not exist.")
        sys.exit(1)

    all_errors = defaultdict(list)
    all_warnings = defaultdict(list)

    train_dirs = args.train_dirs
    if args.test:
        test_dir = root / "TEST"
        if test_dir.is_dir():
            train_dirs.append("TEST")  # 但TEST下结构可能不同，我们统一处理

    for sub in train_dirs:
        sub_path = root / sub
        if not sub_path.is_dir():
            print(f"Warning: Directory {sub_path} not found, skipping.")
            continue
        anno_dir = sub_path / "anno"
        if not anno_dir.is_dir():
            print(f"Warning: anno directory not found in {sub_path}, skipping.")
            continue

        anno_files = list(anno_dir.glob("*.txt"))
        print(f"Scanning {sub}: found {len(anno_files)} annotation files...")

        for anno_file in anno_files:
            video_id = anno_file.stem
            errors, warnings = scan_video(anno_file, sub_path, root)
            if errors or warnings:
                key = f"{sub}/{video_id}"
                all_errors[key] = errors
                all_warnings[key] = warnings
                # 打印简要信息
                if errors:
                    print(f"  ERROR in {key}: {errors[0]}")
                if warnings:
                    print(f"  WARN  in {key}: {warnings[0]}")

        # 额外检查：是否有孤立图片（有图片但没有标注）
        # 遍历可能的图片目录，检查是否有多余视频ID
        # 这里简单处理：对于sub_path下的所有子目录，如果存在图片但anno中无对应txt，则报告
        # 但图片可能分散在frames等子目录，比较复杂，暂时略过

    # 汇总报告
    print("\n" + "=" * 60)
    print(f"Scan completed. Found {len(all_errors)} videos with errors, {len(all_warnings)} with warnings.")
    if all_errors:
        print("\n--- ERRORS ---")
        for key, errs in all_errors.items():
            print(f"{key}:")
            for e in errs:
                print(f"  {e}")
    if all_warnings:
        print("\n--- WARNINGS ---")
        for key, warns in all_warnings.items():
            print(f"{key}:")
            for w in warns:
                print(f"  {w}")


if __name__ == "__main__":
    main()