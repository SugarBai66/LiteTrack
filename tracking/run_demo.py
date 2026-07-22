import cv2
import argparse
from lib.test.evaluation import Tracker


def run_demo(tracker_name='litetrack', cfg_name='vitb_384_mae_ce_32x4_ep300',
             weight_path=None, video_path=None):
    # 构建参数字典
    params = {
        'checkpoint': weight_path,
        'config': cfg_name,
    }
    # 创建 Tracker 实例（'demo' 是任意数据集名称，仅用于占位）
    tracker = Tracker(tracker_name, 'demo', 'demo')

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    # 读取第一帧用于选择目标
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot read first frame")
        return

    # 用户选择初始目标框
    bbox = cv2.selectROI("Select Target", frame, False)
    cv2.destroyWindow("Select Target")
    if bbox == (0, 0, 0, 0):
        print("No target selected, exiting")
        return

    # 调用 run_video 开始跟踪
    # debug=1 会显示跟踪过程可视化（如果内置支持）
    # optional_box 为 (x, y, w, h)
    tracker.run_video(video_path, optional_box=bbox, debug=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='vitb_384_mae_ce_32x4_ep300',
                        help='Model config name')
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to model weight file (.pth)')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    args = parser.parse_args()

    run_demo(cfg_name=args.config, weight_path=args.weight, video_path=args.video)