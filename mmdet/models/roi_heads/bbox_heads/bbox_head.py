import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms, perclass_nms)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


@HEADS.register_module()
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        #TODO: replace 1024 with variable
        self.fisher_box_layer = nn.Parameter(torch.zeros(num_classes, 1024 * 4), requires_grad=False)
        self.grad_box_layer = nn.Parameter(torch.zeros(num_classes, 1024 * 4), requires_grad=False)
        self.alpha = 0.1

    @property
    def use_sigmoid(self):
        return self.loss_cls.use_sigmoid

    @property
    def group_activation(self):
        return getattr(self.loss_cls, 'group', False)

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)

                self.update_representations(losses['loss_bbox'], bbox_pred, labels, pos_inds)
                loss_trans, f_n_i_norm = self.compute_loss_trans(bbox_pred, bbox_targets, labels, pos_inds, avg_factor)
                losses['loss_bbox_trans'] = 0.001 * loss_trans

            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses, f_n_i_norm

    def compute_loss_trans(self, bbox_pred, bbox_targets, labels, pos_inds, avg_factor):

        def class_excluding_softmax(matrix, labels, temperature=1e-1):
            N, C = matrix.size()

            # Create a mask to exclude the ground truth class for each sample
            mask = torch.ones(N, C, dtype=torch.bool).to(matrix.device)
            mask[torch.arange(N), labels] = 0

            # Apply the mask to get the scores for the remaining C-1 classes
            masked_matrix = matrix[mask].view(N, C - 1)

            # Apply temperature scaling
            scaled_matrix = masked_matrix / temperature

            # Compute the softmax over the remaining C-1 classes
            softmax_output = torch.nn.functional.softmax(scaled_matrix, dim=1)

            # Create the final output tensor and fill in the softmax values
            output = torch.zeros(N, C).to(matrix.device)
            output[mask] = softmax_output.view(-1)

            return output

        pos_bbox_pred_all = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool)]
        loss_bbox_full = torch.abs(pos_bbox_pred_all - bbox_targets[pos_inds.type(torch.bool)].unsqueeze(1))
        # loss_bbox_full = loss_bbox_full.sum(-1).sum(-1) / avg_factor

        pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
        loss_bbox = torch.abs(pos_bbox_pred - bbox_targets[pos_inds.type(torch.bool)])
        loss_bbox = loss_bbox.sum(-1) / avg_factor

        F_n = []
        for i in range(len(loss_bbox)):
            grad_persample = torch.autograd.grad(loss_bbox[i], self.fc_reg.weight, retain_graph=True)[0]
            F_n.append(grad_persample.t().view(grad_persample.size(1), -1, 4)[:, labels[pos_inds.type(torch.bool)][i], :].contiguous().view(-1))
        F_n = torch.stack(F_n)
        F_n = F_n**2
        # f_n_i = torch.matmul(F_n, self.fisher_box_layer.transpose(0, 1))
        # f_n_i = class_excluding_softmax(f_n_i, labels[pos_inds.type(torch.bool)])
        # print((f_n_i.max(-1)[0]).mean())
        # print(f_n_i.max())
        F_n_normalized = torch.nn.functional.normalize(F_n, dim=-1)
        fisher_box_layer_normalized = torch.nn.functional.normalize(self.fisher_box_layer.transpose(0, 1), dim=0)
        f_n_i_norm = torch.matmul(F_n_normalized, fisher_box_layer_normalized)
        f_n_i_norm = class_excluding_softmax(f_n_i_norm, labels[pos_inds.type(torch.bool)])
        # print(f_n_i_norm.max())

        updated_fisher_mask = (self.fisher_box_layer.sum(-1)!=0).float().unsqueeze(0)
        loss_trans = updated_fisher_mask.detach() * f_n_i_norm.detach() * loss_bbox_full.sum(-1)

        # loss_trans = torch.ones(loss_bbox_full.shape[:2]).to(loss_bbox_full.device).softmax(-1) * loss_bbox_full.sum(-1)
        return loss_trans.sum(), f_n_i_norm



    def update_grad(self, input_representation, labels):
        with torch.no_grad():  # Disable gradient calculation
            # Initialize a tensor to store the summed representation for each class
            summed_representation = torch.zeros_like(self.fisher_box_layer)

            # Use index_add_ to sum the representations for each class
            summed_representation.index_add_(0, labels, input_representation)

            # Count the number of samples for each class
            counts = torch.bincount(labels, minlength=self.fisher_box_layer.shape[0])

            # Avoid division by zero
            counts[counts == 0] = 1

            # Calculate the average representation for each class
            avg_representation = summed_representation / counts[:, None]

            # Update the Grad
            self.grad_box_layer.mul_(1 - self.alpha).add_(self.alpha * avg_representation)

    def update_fisher(self, input_representation, labels):
        with torch.no_grad():  # Disable gradient calculation
            #compute diagnoal fisher:
            input_representation = input_representation ** 2

            # Initialize a tensor to store the summed representation for each class
            summed_representation = torch.zeros_like(self.fisher_box_layer)

            # Use index_add_ to sum the representations for each class
            summed_representation.index_add_(0, labels, input_representation)

            # Count the number of samples for each class
            counts = torch.bincount(labels, minlength=self.fisher_box_layer.shape[0])

            # Avoid division by zero
            counts[counts == 0] = 1

            # Calculate the average representation for each class
            avg_representation = summed_representation / counts[:, None]

            # Update the Fisher representation
            self.fisher_box_layer.mul_(1 - self.alpha).add_(self.alpha * avg_representation)

    def update_representations(self, loss_bbox, bbox_pred, labels, pos_inds):
        with torch.no_grad():
            grad_w = torch.autograd.grad(loss_bbox, self.fc_reg.weight, retain_graph=True)[0]
            _, C = grad_w.shape
            pos_grad_w = grad_w.t().view(C, -1, 4)[:, labels[pos_inds.type(torch.bool)], :]
            C, pos_N, _ = pos_grad_w.shape
            pos_grad_w = pos_grad_w.transpose(0, 1).reshape(pos_N, -1)

            self.update_grad(pos_grad_w, labels[pos_inds.type(torch.bool)])
            self.update_fisher(pos_grad_w, labels[pos_inds.type(torch.bool)])



        return pos_grad_w

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        if cls_score is not None:
            if self.use_sigmoid:
                if self.group_activation:
                    scores = self.loss_cls.get_activation(cls_score)
                else:
                    scores = F.sigmoid(cls_score)
                    dummpy_prob = scores.new_zeros((scores.size(0), 1))
                    scores = torch.cat([scores, dummpy_prob], dim=1)
            else:
                if self.group_activation:
                    scores = self.loss_cls.get_activation(cls_score)
                else:
                    scores = F.softmax(cls_score, dim=1)
        else:
            scores = None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        elif cfg.get('perclass_nms', False):
            det_bboxes, det_labels = perclass_nms(bboxes, scores,
                                                  cfg.score_thr, cfg.nms,
                                                  cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
