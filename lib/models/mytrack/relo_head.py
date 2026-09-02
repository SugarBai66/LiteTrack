import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
# from mmengine.model import constant_init, normal_init
# from mmcv.cnn import ConvModule
# from mmcv.ops import DeformConv2d
# from .decoder_head import DecoderHead
# from mmcv.ops import
# from mmdet.core.utils import DeformConv

from lib.models.layers.frozen_bn import FrozenBatchNorm2d


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def c_sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.grid_size = self.feat_sz + 1
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=2)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=2)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.grid_size).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.grid_size, 1)) \
                .view((self.grid_size * self.grid_size,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.grid_size)) \
                .view((self.grid_size * self.grid_size,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, torch.stack(
                (prob_vec_tl, prob_vec_br), 1)
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.grid_size * self.grid_size))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, feat_tz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        # self.feat_sz = feat_sz
        # self.feat_tz = feat_tz
        self.stride = stride
        # self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, feat_size, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if True:
            bbox = self.cal_bbox(score_map_ctr, feat_size, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, feat_size, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // feat_size
        idx_x = idx % feat_size

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / feat_size,
                          (idx_y.to(torch.float) + offset[:, 1:]) / feat_size,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset


def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = cfg.MODEL.HEAD.NUM_CHANNELS
    if cfg.MODEL.HEAD.TYPE == "RELO":
        print("==========> 使用 RELOHead! <==========")
        return RELOHead(inplanes=hidden_dim, channel=channel, feat_sz=feat_sz, stride=stride)

    elif "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=hidden_dim, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
        return center_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)


###################################################################################################################
# ---------------------- RELO Head 实现 ----------------------
class PolicyHead(nn.Module):
    """参考 RELO policy_model.py：输出每个位置的 Logit"""

    def __init__(self, inplanes=64, channel=256, freeze_bn=False):
        super().__init__()
        self.conv1_p = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_p = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_p = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_p = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_p = nn.Conv2d(channel // 8, 1, kernel_size=1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, C, H, W)
        x_p1 = self.conv1_p(x)
        x_p2 = self.conv2_p(x_p1)
        x_p3 = self.conv3_p(x_p2)
        x_p4 = self.conv4_p(x_p3)
        logits = self.conv5_p(x_p4)  # (B, 1, H, W)
        return logits.flatten(1)  # (B, H*W)


class ValueHead(nn.Module):
    """参考 RELO value_model.py：输出状态价值 (B, 1)"""

    def __init__(self, inplanes=64, channel=256, freeze_bn=False):
        super().__init__()
        self.conv1_v = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_v = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_v = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_v = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.fc_v = nn.Linear(channel // 8, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, C, H, W)
        x_v1 = self.conv1_v(x)
        x_v2 = self.conv2_v(x_v1)
        x_v3 = self.conv3_v(x_v2)
        x_v4 = self.conv4_v(x_v3)  # (B, C/8, H, W)
        x_v = F.adaptive_avg_pool2d(x_v4, 1).view(x_v4.size(0), -1)
        value = self.fc_v(x_v)  # (B, 1)
        return value.squeeze(-1)  # (B,)


class RELOHead(nn.Module):
    """RELO 集成头：回归 + 策略 + 价值"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(RELOHead, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride

        # 1. 回归头：直接复用已有的 CenterPredictor（它内部已预测 size 和 offset）
        self.reg_head = CenterPredictor(inplanes=inplanes, channel=channel,
                                        feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn)
        # 2. 策略头
        self.policy_head = PolicyHead(inplanes=inplanes, channel=channel, freeze_bn=freeze_bn)
        # 3. 价值头
        self.value_head = ValueHead(inplanes=inplanes, channel=channel, freeze_bn=freeze_bn)

    def forward(self, x, feat_size):
        """
        输入: x (B, C, H, W)
        输出:
            pred_boxes: (B, 4)        取策略分数最高位置的框，用于现有 GIoU/L1 损失
            all_boxes:  (B, H*W, 4)   每个位置的框，用于后续 RL 采样
            policy_logits: (B, H*W)   每个位置的 Logit，用于 RL 训练
            values: (B,)              状态价值，用于 Actor-Critic
        """
        # 1. 回归头：获取所有位置的 size 和 offset
        # 注意：CenterPredictor.forward 会返回 score_map, bbox, size_map, offset_map
        # 其中 bbox 是基于 score_map_ctr 的 argmax 得到的（这里我们不直接用它的 bbox，而是自己算）
        score_map_ctr, _, size_map, offset_map = self.reg_head(x, feat_size)

        # 2. 策略头：获取每个位置的 Logit
        policy_logits = self.policy_head(x)  # (B, H*W)

        # 3. 价值头：获取状态价值
        values = self.value_head(x)  # (B,)

        # 4. 核心：根据策略分数（policy_logits）的 argmax 来选取对应位置的框
        # 这里我们取 logits 最大的索引，对应到 size_map 和 offset_map 上，得到最终 bbox
        max_idx = policy_logits.argmax(dim=1, keepdim=True)  # (B, 1)

        # 将 idx 扩展成 (B, 2, 1) 用于 size 和 offset 的 gather
        idx_2d = max_idx.unsqueeze(1).expand(-1, 2, 1)  # (B, 2, 1)

        # 从 size_map 和 offset_map 中选出对应位置的数值
        # size_map: (B, 2, H, W) -> flatten(2) -> (B, 2, H*W) -> gather -> (B, 2, 1)
        size_selected = size_map.flatten(2).gather(dim=2, index=idx_2d).squeeze(-1)  # (B, 2)
        offset_selected = offset_map.flatten(2).gather(dim=2, index=idx_2d).squeeze(-1)  # (B, 2)

        # 计算选中的位置坐标 (cx, cy)
        idx_y = max_idx // feat_size
        idx_x = max_idx % feat_size
        cx = (idx_x.float() + offset_selected[:, :1]) / feat_size
        cy = (idx_y.float() + offset_selected[:, 1:]) / feat_size

        # 拼接成最终 pred_boxes (cx, cy, w, h) -> (B, 4)
        pred_boxes = torch.cat([cx, cy, size_selected], dim=1)
        pred_boxes = pred_boxes.unsqueeze(1)  # 变为 (B,1,4)

        # 5. 计算所有位置的框（用于后续 RL 采样或分析）
        # 这里我们可以直接用 size_map 和 offset_map 平铺出来
        # 需要构造一个网格坐标
        H, W = size_map.shape[2], size_map.shape[3]
        # 新代码：去掉 unsqueeze，直接使用 1D 张量
        x_coords = torch.arange(W, device=x.device, dtype=torch.float32) / feat_size  # shape: (W,)
        y_coords = torch.arange(H, device=x.device, dtype=torch.float32) / feat_size  # shape: (H,)

        # 使用 indexing='ij' 确保生成 (H, W) 的网格
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_x = grid_x.reshape(-1)  # 展平为 (H*W,)
        grid_y = grid_y.reshape(-1)

        # size_map 和 offset_map 的 shape 是 (B, 2, H, W)
        # flatten(2) 后变成 (B, 2, H*W)，我们需要转置一下
        all_sizes = size_map.flatten(2).permute(0, 2, 1)  # (B, H*W, 2)
        all_offsets = offset_map.flatten(2).permute(0, 2, 1)  # (B, H*W, 2)

        # 计算所有框 (cx, cy, w, h)
        all_cx = grid_x.unsqueeze(0) + all_offsets[:, :, 0:1] / feat_size
        all_cy = grid_y.unsqueeze(0) + all_offsets[:, :, 1:2] / feat_size
        all_boxes = torch.cat([all_cx, all_cy, all_sizes], dim=-1)  # (B, H*W, 4)

        # 返回字典，兼容现有 actor 的损失计算
        return {
            'pred_boxes': pred_boxes,
            'score_map': score_map_ctr,  # 原始 center 得分图（可能用于分类损失）
            'size_map': size_map,
            'offset_map': offset_map,
            'policy_logits': policy_logits,
            'values': values,
            'all_boxes': all_boxes,
        }

    def cal_bbox(self, score_map, feat_size, size_map, offset_map):
        return self.reg_head.cal_bbox(score_map, feat_size, size_map, offset_map)


# # 修改 build_box_head，增加 RELO 分支
# def build_box_head(cfg, hidden_dim):
#     stride = cfg.MODEL.BACKBONE.STRIDE
#     feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
#     channel = cfg.MODEL.HEAD.NUM_CHANNELS
#
#     if cfg.MODEL.HEAD.TYPE == "RELO":
#         print("==========> 使用 RELOHead! <==========")
#         return RELOHead(inplanes=hidden_dim, channel=channel, feat_sz=feat_sz, stride=stride)
#
#     elif "CORNER" in cfg.MODEL.HEAD.TYPE:
#         # ... (原有代码保持不变)
#         pass
#     elif cfg.MODEL.HEAD.TYPE == "CENTER":
#         # ... (原有代码保持不变)
#         pass
#     else:
#         raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)