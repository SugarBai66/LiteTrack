import numpy as np
import os
import shutil
import argparse
import _init_paths
from lib.test.evaluation.environment import env_settings


def transform_trackingnet(tracker_name, cfg_name):
    env = env_settings()
    result_dir = env.results_path
    src_dir = os.path.join(result_dir, "%s/%s/trackingnet/" % (tracker_name, cfg_name))
    dest_dir = os.path.join(result_dir, "%s/%s/trackingnet_submit/" % (tracker_name, cfg_name))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    items = os.listdir(src_dir)
    for item in items:
        if "all" in item:
            continue
        if "time" not in item:
            src_path = os.path.join(src_dir, item)
            dest_path = os.path.join(dest_dir, item)
            bbox_arr = np.loadtxt(src_path, dtype=np.int, delimiter='\t')
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')

    src_zip_path = os.path.join(os.path.dirname(src_dir.rstrip('/')),
                                os.path.basename(src_dir.rstrip('/')))
    dest_zip_path = os.path.join(os.path.dirname(dest_dir.rstrip('/')),
                                 os.path.basename(dest_dir.rstrip('/')))
    shutil.make_archive(src_zip_path, "zip", src_dir)
    shutil.make_archive(dest_zip_path, "zip", dest_dir)
    print(f"Zip saved to: {src_zip_path}.zip")
    print(f"Zip saved to: {dest_zip_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transform trackingnet results.')
    parser.add_argument('--tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('--cfg_name', type=str, help='Name of config file.')

    args = parser.parse_args()
    transform_trackingnet(args.tracker_name, args.cfg_name)
