import numpy as np
import pandas as pd


def load_text_numpy(path, delimiter, dtype):
    # 首先尝试用 np.loadtxt 快速读取（适用于纯数字且列数一致）
    try:
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype)
    except Exception:
        # 如果失败，手动逐行解析，只提取前 4 个数字（x, y, w, h）
        rows = []
        with open(path, 'r') as f:
            for line in f:
                # 按空白分割（无论 delimiter 是什么，都用 split() 分割）
                tokens = line.strip().split()
                # 收集所有能转换为数字的 token
                nums = []
                for t in tokens:
                    try:
                        nums.append(dtype(t))
                    except ValueError:
                        # 忽略非数字（如字符串）
                        pass
                # 确保至少有 4 个数字，取前 4 个作为矩形框
                if len(nums) >= 4:
                    rows.append(nums[:4])
                # 否则跳过该行（或可打印警告）
        if not rows:
            raise Exception('No valid rows found in file {}'.format(path))
        return np.array(rows, dtype=dtype)


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str
