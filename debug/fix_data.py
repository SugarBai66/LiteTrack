import os
import numpy as np

def check_file(filepath):
    try:
        # 尝试用 np.loadtxt 读取
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # 检查是否有空行或形状异常
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != 4:
            return False, f"Wrong columns: {data.shape[1]}"
        return True, None
    except Exception as e:
        return False, str(e)

broken_files = []
root = "/mnt/ssd4t/datasets/TrackingNet"
for train_dir in [f"TRAIN_{i}" for i in range(12)]:
    anno_dir = os.path.join(root, train_dir, "anno")
    if not os.path.exists(anno_dir):
        continue
    for fname in os.listdir(anno_dir):
        if fname.endswith(".txt"):
            fpath = os.path.join(anno_dir, fname)
            ok, err = check_file(fpath)
            if not ok:
                broken_files.append((fpath, err))

print(f"Found {len(broken_files)} broken files:")
for path, err in broken_files:
    print(f"{path}: {err}")