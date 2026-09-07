# lib/train/actors/litetrack_relo_actor.py

from .base_actor import BaseActor
import torch

class LiteTrackWarmupActor(BaseActor):
    """LiteTrack 预热训练器（回归头热身）"""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

    def __call__(self, data):
        out_list = self.forward_pass(data)
        loss, status = self.compute_losses(out_list, data)
        return loss, status

    def forward_pass(self, data):
        """
        data 中包含：
            template_images: list of tensors, 每个 shape (B,3,H_t,W_t)
            search_images:   list of tensors, 每个 shape (B,3,H_s,W_s)
            template_anno:   list of tensors, 每个 shape (B,4)
            search_anno:     tensor, shape (B,4)
        """
        # 取第一个模板和第一个搜索帧（单帧训练）
        template = data['template_images'][0]   # (B,3,H_t,W_t)
        search = data['search_images'][0]       # (B,3,H_s,W_s)

        # 直接调用 LiteTrack 的前向
        out_dict = self.net(template, search)   # 返回包含 pred_boxes, score_map, size_map, offset_map 的字典

        # 包装成列表（兼容后续 compute_losses 的循环）
        return [out_dict]

    def compute_losses(self, pred_list, gt_dict, return_status=True):
        """复用 RELO warmup 的损失计算逻辑"""
        from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
        from lib.utils.heapmap_utils import generate_heatmap

        device = pred_list[0]['pred_boxes'].device
        total_loss = torch.tensor(0.0, device=device)
        total_status = {}

        # 生成高斯热图作为中心监督
        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.ENCODER.STRIDE
        )

        for i, pred in enumerate(pred_list):
            gt_bbox = gt_dict['search_anno'][i]
            gt_gaussian_map = gt_gaussian_maps[i].unsqueeze(1)

            pred_boxes = pred['pred_boxes']  # (B, 1, 4)
            # 适配维度，确保和 gt 对齐
            if pred_boxes.dim() == 3:
                pred_boxes = pred_boxes.view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(0.0, 1.0)

            # GIoU 损失
            giou_loss, iou = self.objective['giou'](pred_boxes, gt_boxes_vec)
            l1_loss = self.objective['l1'](pred_boxes, gt_boxes_vec)

            # Focal 损失（中心定位）
            if 'score_map' in pred:
                location_loss = self.objective['focal'](pred['score_map'], gt_gaussian_map)
            else:
                location_loss = torch.tensor(0.0, device=device)

            loss = (self.loss_weight['giou'] * giou_loss +
                    self.loss_weight['l1'] * l1_loss +
                    self.loss_weight['focal'] * location_loss)

            total_loss += loss

            if return_status:
                total_status[f"Frame{i}/GIoU"] = giou_loss.item()
                total_status[f"Frame{i}/L1"] = l1_loss.item()
                total_status[f"Frame{i}/Focal"] = location_loss.item()

        total_loss = total_loss / len(pred_list)
        if return_status:
            total_status["Loss/Total"] = total_loss.item()
        return total_loss, total_status


# lib/train/actors/litetrack_relo_actor.py (续)

class LiteTrackRELOActor(BaseActor):
    """LiteTrack 强化学习训练器"""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg
        self.auc_r_weight = cfg.TRAIN.AUC_REWARD_WEIGHT
        self.iou_r_weight = cfg.TRAIN.IOU_REWARD_WEIGHT
        self.adv_norm = cfg.TRAIN.ADV_NORM

    def __call__(self, data):
        out_list = self.forward_pass(data)
        loss, status = self.compute_losses(out_list, data)
        return loss, status

    def forward_pass(self, data):
        """单帧前向，直接调用 LiteTrack"""
        template = data['template_images'][0]
        search = data['search_images'][0]
        out_dict = self.net(template, search)
        return [out_dict]

    def compute_losses(self, pred_list, gt_dict, return_status=True):
        """RELO 强化学习损失：采样 + IoU奖励 + AUC奖励 + Actor-Critic"""
        import torch
        from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy

        device = pred_list[0]['policy_logits'].device
        total_status = {}

        logp_list = []
        reward_list = []
        iou_sampling_list = []
        value_list = []

        for i, pred in enumerate(pred_list):
            gt_bbox = gt_dict['search_anno'][i]
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(0.0, 1.0)

            policy_logits = pred['policy_logits']   # (B, H*W)
            value = pred['value']                   # (B,)
            size_map = pred['size_map']             # (B, 2, H, W)
            offset_map = pred['offset_map']         # (B, 2, H, W)

            # 1. 按策略分布采样一个位置
            dist = torch.distributions.Categorical(logits=policy_logits)
            action = dist.sample()                  # (B,)
            log_prob = dist.log_prob(action)        # (B,)

            # 2. 取出该位置对应的框
            bbox_sampling = self.net.box_head.cal_bbox_sampling(
                idx=action,
                size_map=size_map,
                offset_map=offset_map
            )  # (B, 4) cx,cy,w,h
            bbox_sampling_vec = box_cxcywh_to_xyxy(bbox_sampling).view(-1, 4)

            # 3. 计算帧级 IoU 奖励
            _, iou_reward = self.objective['giou'](bbox_sampling_vec, gt_boxes_vec)

            iou_sampling_list.append(iou_reward.detach())
            reward_list.append(self.iou_r_weight * iou_reward.detach())
            logp_list.append(log_prob)
            value_list.append(value)

            if return_status:
                total_status[f"Frame{i}/IoU_reward"] = iou_reward.detach().mean().item()

        # 4. 计算序列级 AUC 奖励（这里是单帧，但保留接口）
        # 注意：这里 iou_sampling_list 长度如果为 1，AUC 退化为 IoU 本身
        auc_reward = self.compute_auc(iou_sampling_list)
        reward_list = [r + auc_reward * self.auc_r_weight for r in reward_list]

        # 5. Actor-Critic 损失
        rewards = torch.stack(reward_list)
        values = torch.stack(value_list)
        logps = torch.stack(logp_list)

        advantages = rewards - values
        if self.adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(logps * advantages.detach()).mean()
        value_loss = ((values - rewards.detach()) ** 2).mean()
        rl_loss = policy_loss + 0.5 * value_loss
        total_loss = rl_loss * self.loss_weight['rl']

        if return_status:
            total_status["AUC_reward"] = auc_reward.detach().mean().item()
            total_status["Loss/Policy"] = policy_loss.item()
            total_status["Loss/Value"] = value_loss.item()
            total_status["Loss/RL"] = rl_loss.item()
            total_status["Loss/Total"] = total_loss.item()

        return total_loss, total_status

    def compute_auc(self, seq_ious, num_thresholds=21):
        """计算 AUC 奖励（即使单帧也可用）"""
        import torch
        iou_mat = torch.stack(seq_ious, dim=1)  # (B, T)
        thresholds = torch.linspace(0, 1, num_thresholds, device=iou_mat.device)
        iou_expanded = iou_mat[:, None, :]          # (B, 1, T)
        threshold_expanded = thresholds[None, :, None]  # (1, K, 1)
        success_mat = (iou_expanded > threshold_expanded).float()
        success_rate = success_mat.mean(dim=2)      # (B, K)
        auc_per_sample = torch.trapz(success_rate, thresholds, dim=1)
        return auc_per_sample.detach()