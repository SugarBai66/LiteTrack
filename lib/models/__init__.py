# 默认只有这两行
# from .ostrack.ostrack import build_ostrack
# from .litetrack.litetrack import build_LiteTrack

# 1. 必须保留原来的 ostrack 导入（这是报错缺失的部分）
from .ostrack.ostrack import build_ostrack

# 2. 引入 liteTrack 的两个版本
from .litetrack.litetrack import build_LiteTrack as build_LiteTrack_orig
from .mytrack.litetrack import build_LiteTrack as build_LiteTrack_mine

# 3. 定义统一的入口函数（方案一的核心）
def build_LiteTrack(cfg, training=True):
    if cfg.MODEL.get('USE_MYTRACK', False):
        print("==========> 使用 mytrack 代码! <==========")
        return build_LiteTrack_mine(cfg)
    else:
        print("==========> 使用官方 litetrack 代码! <==========")
        return build_LiteTrack_orig(cfg)