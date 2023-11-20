import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.ops import Conv2d
from mmcv.ops.carafe import CARAFEPack
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from mmdet.models.builder import HEADS, build_loss

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

        #TODO: replace 1024 with variable
        self.fisher_box_layer = nn.Parameter(torch.zeros(num_classes, 256), requires_grad=False)
        self.grad_box_layer = nn.Parameter(torch.zeros(num_classes, 256), requires_grad=False)
        self.alpha = 0.1

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def freeze(self, cfg):
        if cfg['type'] == 'all':
            for p in self.parameters():
                p.requires_grad = False
        elif cfg['type'] == 'feat':
            for name, p in self.named_parameters():
                if "conv_logits" not in name:
                    p.requires_grad = False
        elif cfg['type'] == 'none':
            pass
        else:
            raise NotImplementedError


    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels, f_n_i_norm):
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum() * 0
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask

        pos_inds=None
        avg_factor=1.
        # self.update_representations(loss['loss_mask'], mask_pred, labels, pos_inds)
        loss_trans = self.compute_loss_trans(mask_pred, mask_targets, labels, pos_inds, avg_factor, f_n_i_norm)
        loss['loss_mask_trans'] = 0.01 * loss_trans

        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].cpu().numpy())
        return cls_segms

    def compute_loss_trans(self, mask_pred, mask_targets, labels, pos_inds, avg_factor, f_n_i_norm):

        def class_excluding_softmax(matrix, labels, temperature=1.0):
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

        loss_mask_full = F.binary_cross_entropy_with_logits(mask_pred, mask_targets.unsqueeze(1).repeat(1, mask_pred.size(1), 1, 1), reduction="none")
        loss_mask_full = loss_mask_full.mean(-1).mean(-1)

        # num_rois = mask_pred.size()[0]
        # inds = torch.arange(0, num_rois, dtype=torch.long, device=mask_pred.device)
        # pred_slice = mask_pred[inds, labels].squeeze(1)
        # loss_mask = F.binary_cross_entropy_with_logits(pred_slice, mask_targets, reduction='none')
        # loss_mask = loss_mask.mean(-1).mean(-1)
        #
        # F_n = []
        # for i in range(len(loss_mask)):
        #     grad_persample = torch.autograd.grad(loss_mask[i], self.conv_logits.weight, retain_graph=True)[0]
        #     F_n.append(grad_persample[labels[i]].squeeze(-1).squeeze(-1))
        # F_n = torch.stack(F_n)
        # F_n = F_n ** 2
        # # f_n_i = torch.matmul(F_n, self.fisher_box_layer.transpose(0, 1))
        # # f_n_i = class_excluding_softmax(f_n_i, labels)
        #
        # F_n_normalized = torch.nn.functional.normalize(F_n, dim=-1)
        # fisher_box_layer_normalized = torch.nn.functional.normalize(self.fisher_box_layer.transpose(0, 1), dim=0)
        # f_n_i_norm = torch.matmul(F_n_normalized, fisher_box_layer_normalized)
        # f_n_i_norm = class_excluding_softmax(f_n_i_norm, labels)
        #
        # updated_fisher_mask = (self.fisher_box_layer.sum(-1)!=0).float().unsqueeze(0)
        # loss_trans = updated_fisher_mask.detach() * f_n_i_norm.detach() * loss_mask_full

        loss_trans = f_n_i_norm * loss_mask_full
        return loss_trans.sum()
        # return loss_mask_full.mean(-1).sum()

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
            grad_w = torch.autograd.grad(loss_bbox, self.conv_logits.weight, retain_graph=True)[0]
            grad_w = grad_w[:, :, 0, 0]
            pos_grad_w = grad_w[labels]

            # self.update_grad(pos_grad_w, labels)
            self.update_fisher(pos_grad_w, labels)

        return pos_grad_w

def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
