import math

from lib.models import build_LiteTrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box, box_xyxy_to_cxcywh, box_xyxy_to_xywh, box_xywh_to_xyxy, box_cxcywh_to_xywh
from lib.utils.box_ops import box_xywh_to_xyxy, box_iou



class LiteTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(LiteTrack, self).__init__(params)
        network = build_LiteTrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        # self.use_visdom = 1
        self.frame_id = 0
        if self.debug:
            # if not self.use_visdom:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            # else:
                # self.add_hook()
                # self._init_visdom(None, 1)
                # pass
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # 【新增】初始化历史记录列表
        self.score_history = []  # 存每帧的最高响应分数
        self.iou_history = []    # 存每帧与真值的IoU（如果有GT的话）
        self.frame_ids = []      # 存帧序号用于X轴

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        self.box_mask_z = None
        
        template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                    template.tensors.device).squeeze(1)
        template_bbox = box_xywh_to_xyxy(template_bbox).float()

        with torch.no_grad():
            self.z_dict1 = template
            self.z_feat = self.network.forward_z(template.tensors, template_bb=template_bbox)
            # self.z_dict1 = self.network.forward_template(template.tensors, self.template_target_box_mask_z)

        # save states
        self.state = info['init_bbox']
        
        # self.template_bbbox = torch.ones_like(template_bbox).float().cuda()
        self.frame_id = 0
        self.last_update_frame = self.frame_id
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, vis=None):
        if self.cfg.MODEL.HEAD.TYPE == 'DECODER':
            return self.track_decoder(image, info=info, vis=vis)
        elif self.cfg.MODEL.HEAD.TYPE == 'GFL':
            return self.track_GFL(image, info=info, vis=vis)
        elif self.cfg.MODEL.HEAD.TYPE == 'CENTER':
            return self.track_center(image, info=info, vis=vis)

    def track_center(self, image, info: dict = None, vis=None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            # out_dict = self.network.forward(
            #     template=self.z_dict1.tensors, search=x_dict.tensors, template_bb=self.template_bbox)
            out_dict = self.network(
                template_feats=self.z_feat, search=x_dict.tensors)    

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, int(self.feat_sz),  out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # 1. 计算当前帧的置信度（取分数图的最大值）
        max_score = pred_score_map.max().item()

        # 2. 如果有 GT 真值框，可以计算 IoU（用于评估性能）
        if 'gt_bbox' in info:
            gt_box = info['gt_bbox']  # list [x, y, w, h]
            pred_box = self.state  # list [x, y, w, h]

            # 转为 Tensor，并增加 batch 维度 -> (1, 4)
            gt_tensor = torch.tensor(gt_box, dtype=torch.float32).unsqueeze(0)
            pred_tensor = torch.tensor(pred_box, dtype=torch.float32).unsqueeze(0)

            # 转换为 xyxy 格式（box_iou 要求）
            gt_xyxy = box_xywh_to_xyxy(gt_tensor)
            pred_xyxy = box_xywh_to_xyxy(pred_tensor)

            # 计算 IoU，返回 (iou, union)，取第一个元素
            iou_val, _ = box_iou(pred_xyxy, gt_xyxy)
            iou_value = iou_val.item()  # 标量
        else:
            iou_value = 0.0

        # 3. 记录到历史列表中
        self.frame_ids.append(self.frame_id)
        MAX_POINTS = 100
        self.score_history.append(max_score)
        if len(self.score_history) > MAX_POINTS:
            self.score_history.pop(0)  # 移除最早的点
            self.frame_ids.pop(0)
        self.iou_history.append(iou_value)

        # 当前帧的 FPS 和内存占用
        debug_info = {
            'Frame_ID': self.frame_id,
            'Target_Size': f'{self.state[2]:.1f} x {self.state[3]:.1f}',
            'Score_Max': float(pred_score_map.max().cpu()),
        }

        if vis is not None:
            print(f"Frame {self.frame_id}: Sending {len(self.score_history)} points to Visdom")
            # 注册所有 visdom 内容
            vis.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking', caption='LiteTrack Tracking')
            # 修改 permute(1,0,1) → permute(2,0,1)
            vis.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region',
                         caption='Search Region')
            vis.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template',
                         caption='Template')

            vis.register(debug_info, 'info_dict', 1, 'Debug_Info')
            # 在你的 litetrack.py 中注册折线图时，窗口名加上帧号
            # 改之后（合并为一个窗口）：
            vis.register(torch.tensor(self.score_history), 'lineplot', 1, 'Confidence_Curve' )

            # 如果有GT，绘制IoU曲线
            if 'gt_bbox' in info:
                if len(self.iou_history) > 0:
                    vis.register(torch.tensor(self.iou_history), 'lineplot', 1, 'IoU_Curve',
                                 caption='IoU per Frame', opts={'xlabel': 'Frame', 'ylabel': 'IoU'})
                # vis.register(
                #     (torch.tensor(self.iou_history), torch.tensor(self.frame_ids)),
                #     'lineplot',
                #     1,
                #     'IoU_Curve',
                #     caption='Tracking IoU per Frame',
                #     opts={'xlabel': 'Frame', 'ylabel': 'IoU', 'legend': ['IoU'],'update': 'replace'}
                # )
            # vis.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map',
                         # caption='Score Map',opts={'width': 40, 'height': 40, 'colormap': 'Jet'}   )
            # 调整宽高)
            # vis.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1,
            #              'score_map_hann', caption='Hann Score', opts={'width': 40, 'height': 40, 'colormap': 'Jet'} )  # 调整宽高

        else:
            # 如果需要保存图片，则在这里实现（可复用原来保存图片的代码）
            if self.debug:
                # 保存图片...
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)

        # for debug
        # if self.debug:
        #     if not self.use_visdom:
        #         x1, y1, w, h = self.state
        #         image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #         cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        #         save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
        #         cv2.imwrite(save_path, image_BGR)
        #     else:
        #         vis.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 0, 'Tracking')
        #
        #         vis.register(torch.from_numpy(x_patch_arr).permute(1, 0, 1), 'image', 1, 'search_region')
        #         vis.register(torch.from_numpy(self.z_patch_arr).permute(1, 0, 1), 'image', 1, 'template')
        #         vis.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 0, 'score_map')
        #         vis.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 0, 'score_map_hann')

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return LiteTrack
