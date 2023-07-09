import math
from scipy.optimize import linear_sum_assignment

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import torchvision

from models.transformer import build_transformer
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x.unsqueeze(-1)


class DETRHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, use_interaction_decoder=False, x_improved=False, cat_specific_fc=False, verb_embed_norm=False, interaction_decoder=None, use_image_verb_loss=False, ffn=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.x_improved = x_improved
        if x_improved:
            self.roi_ho_downsample = nn.Linear(hidden_dim * 2, hidden_dim)
            self.roi_ho_dropout = nn.Dropout(0.1)
            self.ho_feat_norm = nn.LayerNorm(hidden_dim)
            self.conv_feature_map = nn.Conv2d(hidden_dim * 2, hidden_dim,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=True)
        self.use_interaction_decoder = use_interaction_decoder
        if use_interaction_decoder:
            assert interaction_decoder is not None, "interaction_decoder should be provided"
            self.interaction_decoder = interaction_decoder
            # self.verb_class_embed = nn.Linear(hidden_dim, hidden_dim)  # is one layer enough?
            i_hidden_dim = interaction_decoder.d_model
            if i_hidden_dim != hidden_dim:
                self.interaction_feature_proj = nn.Conv2d(hidden_dim, i_hidden_dim, kernel_size=1)
                self.interaction_pos_proj = nn.Conv2d(hidden_dim, i_hidden_dim, kernel_size=1)
                self.verb_class_embed = nn.Sequential(
                    nn.Linear(hidden_dim, i_hidden_dim),
                    nn.LayerNorm(i_hidden_dim),
                )
            else:
                self.interaction_feature_proj = nn.Identity()
                self.interaction_pos_proj = nn.Identity()
                self.verb_class_embed = nn.Identity()
            self.interaction_tgt_embed = nn.Embedding(num_verb_classes, i_hidden_dim)
            self.interaction_query_embed = nn.Embedding(num_verb_classes, i_hidden_dim)
            self.interaction_prototype_embed = nn.Identity()
            self.use_image_verb_loss = use_image_verb_loss
            self.interaction_decoder_layers = self.interaction_decoder.num_decoder_layers
            if use_image_verb_loss:
                # self.interaction_class_embed = nn.Linear(hidden_dim, 1)  # binary classification for each interaction query
                if cat_specific_fc:
                    self.interaction_class_embed = nn.Sequential(ffn, GroupWiseLinear(num_verb_classes, i_hidden_dim, bias=True))
                else:
                    self.interaction_class_embed = nn.Sequential(ffn, nn.Linear(i_hidden_dim, 1))
            # hard residual
            self.verb_class_weight = nn.Parameter(torch.Tensor(num_verb_classes, i_hidden_dim))
            self.verb_embed_norm = verb_embed_norm
            if verb_embed_norm:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        # category query tgt
        # layer = self.interaction_tgt_embed
        # weight_init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # # if layer.bias is not None:
        # #     fan_in, _ = weight_init._calculate_fan_in_and_fan_out(layer.weight)
        # #     bound = 1 / math.sqrt(fan_in)
        # #     weight_init.uniform_(layer.bias, -bound, bound)

        # verb class static weight
        weight_init.kaiming_uniform_(self.verb_class_weight, a=math.sqrt(5))

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        proj_src = self.input_proj(src)
        hs, memory = self.transformer(proj_src, mask, self.query_embed.weight, pos[-1])

        original_mask = samples.mask
        original_hs = original_mask.shape[1] - original_mask[:, :, 0].sum(dim=-1)
        original_ws = original_mask.shape[2] - original_mask[:, 0, :].sum(dim=-1)
        original_shapes = torch.stack([original_ws, original_hs], dim=-1).to(src)

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()

        ho_embed = hs
        if self.x_improved:
            feature_map = self.conv_feature_map(torch.cat([proj_src, memory], dim=1)) + pos[-1]  # encoder or backbone
            n_layer, bs, _, _ = hs.shape
            human_refs = outputs_sub_coord * original_shapes.unsqueeze(1).repeat([1, 1, 2]) / 32.0  # [n_layer, B, K, 4]
            object_refs = outputs_obj_coord * original_shapes.unsqueeze(1).repeat([1, 1, 2]) / 32.0
            h_ref_list = list(box_cxcywh_to_xyxy(human_refs.transpose(0, 1).flatten(1, 2)))
            o_ref_list = list(box_cxcywh_to_xyxy(object_refs.transpose(0, 1).flatten(1, 2)))
            human_roi = torchvision.ops.roi_pool(feature_map, h_ref_list, output_size=(7, 7), spatial_scale=1.0)
            object_roi = torchvision.ops.roi_pool(feature_map, o_ref_list, output_size=(7, 7), spatial_scale=1.0)
            human_roi = human_roi.view(bs, n_layer, self.num_queries, self.hidden_dim, 7*7).mean(dim=-1).transpose(0, 1)
            object_roi = object_roi.view(bs, n_layer, self.num_queries, self.hidden_dim, 7*7).mean(dim=-1).transpose(0, 1)
            roi_ho_embed = self.roi_ho_dropout(self.roi_ho_downsample(torch.cat([human_roi, object_roi], dim=-1)))
            ho_embed = self.ho_feat_norm(hs + roi_ho_embed)

        if self.use_interaction_decoder:
            i_hs = self.interaction_decoder(self.interaction_feature_proj(memory), mask, self.interaction_tgt_embed.weight, self.interaction_query_embed.weight, self.interaction_pos_proj(pos[-1]))
            i_prototype = self.interaction_prototype_embed(i_hs) / 32
            ho_embed = self.verb_class_embed(ho_embed)
            i_weight = i_prototype.transpose(2, 3) + self.verb_class_weight.T
            if self.verb_embed_norm:
                ho_embed = ho_embed / ho_embed.norm(dim=-1, keepdim=True)
                i_weight = i_weight / i_weight.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
            else:
                logit_scale = 1.0
            all_outputs_verb_class = logit_scale * torch.matmul(ho_embed, i_weight.unsqueeze(1))
            outputs_verb_class = all_outputs_verb_class[-1]
            other_outputs_verb_class = None
            if self.interaction_decoder_layers > 1:
                other_outputs_verb_class = all_outputs_verb_class[:-1]
        else:
            outputs_verb_class = self.verb_class_embed(ho_embed)
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.use_interaction_decoder and self.interaction_decoder_layers > 1:
            i_aux_out = {'pred_verb_logits_i{}'.format(idx): item[-1] for idx, item in enumerate(other_outputs_verb_class)}
            out.update(i_aux_out)
        if self.use_interaction_decoder and self.use_image_verb_loss:
            i_class = self.interaction_class_embed(i_hs)  # image-level supervision
            out.update({'pred_image_verb_logits': i_class[-1].squeeze(-1)})
            if self.interaction_decoder_layers > 1:
                out.update({'pred_image_verb_logits_i{}'.format(idx): item.squeeze(-1) for idx, item in enumerate(i_class[:-1])})
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord, other_outputs_verb_class)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, other_outputs_verb_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.interaction_decoder_layers > 1 and other_outputs_verb_class is not None:
            aux_out = []
            for i in range(len(outputs_obj_class[:-1])):
                layer_aux_out = {'pred_obj_logits': outputs_obj_class[i], 'pred_verb_logits': outputs_verb_class[i], 'pred_sub_boxes': outputs_sub_coord[i], 'pred_obj_boxes': outputs_obj_coord[i]}
                layer_aux_out.update({'pred_verb_logits_i{}'.format(idx): item[i] for idx, item in enumerate(other_outputs_verb_class)})
                aux_out.append(layer_aux_out)
            return aux_out
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, verb_loss_type="focal", image_verb_loss_type="focal", i_dec_layers=1):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type
        self.i_dec_layers = i_dec_layers
        self.image_verb_loss_type = image_verb_loss_type

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        losses = {}
        for i in range(-1, self.i_dec_layers - 1):
            k = 'pred_verb_logits' if i < 0 else 'pred_verb_logits_i{}'.format(i)
            loss_k = 'loss_verb_ce' if i < 0 else 'loss_verb_ce_i{}'.format(i)
            src_logits = outputs[k]

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.zeros_like(src_logits)
            target_classes[idx] = target_classes_o

            if self.verb_loss_type == 'bce':
                pos_weight = 10. * torch.ones(self.num_verb_classes).to(src_logits.device)
                loss_verb_ce_none = F.binary_cross_entropy_with_logits(src_logits, target_classes, pos_weight=pos_weight, reduction='none')
                loss_verb_ce = loss_verb_ce_none.sum(-1).mean()
            elif self.verb_loss_type == 'focal':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._neg_loss(src_logits, target_classes)
            elif self.verb_loss_type == 'asl':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self.asymmetric_loss(src_logits, target_classes)

            losses[loss_k] = loss_verb_ce
        return losses

    def loss_image_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_image_verb_logits' in outputs
        losses = {}
        for i in range(-1, self.i_dec_layers - 1):
            k = 'pred_image_verb_logits' if i < 0 else 'pred_image_verb_logits_i{}'.format(i)
            loss_k = 'loss_image_verb_ce' if i < 0 else 'loss_image_verb_ce_i{}'.format(i)
            src_logits = outputs[k]
            target_classes = torch.stack([t['image_verb_labels'] for t in targets])
            if self.image_verb_loss_type == 'bce':
                pos_weight = 10. * torch.ones(self.num_verb_classes).to(src_logits.device)
                loss_image_verb_ce_none = 0.1 * F.binary_cross_entropy_with_logits(src_logits, target_classes, pos_weight=pos_weight, reduction='none')
                loss_image_verb_ce = loss_image_verb_ce_none.sum(-1).mean()
            elif self.image_verb_loss_type == 'focal':
                src_logits = src_logits.sigmoid()
                loss_image_verb_ce = self._neg_loss(src_logits, target_classes)
            elif self.image_verb_loss_type == 'asl':
                src_logits = src_logits.sigmoid()
                loss_image_verb_ce = self.asymmetric_loss(src_logits, target_classes)

            losses[loss_k] = loss_image_verb_ce
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def asymmetric_loss(self, x, y, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        pos_inds = y.eq(1).float()
        num_pos  = pos_inds.float().sum()

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if clip is not None and clip > 0:
            xs_neg = (xs_neg + clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if gamma_neg > 0 or gamma_pos > 0:
            if disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if num_pos == 0:
            return -loss.sum()
        else:
            return -loss.sum() / num_pos

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'image_verb_labels': self.loss_image_verb_labels,
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        if "pred_image_verb_logits" in outputs:
            losses.update(self.get_loss("image_verb_labels", outputs, targets, indices, num_interactions))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            scores = torch.cat((torch.ones_like(os), os))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})
            results[-1].update({'scores': scores.to('cpu')})

            vs = vs * os.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results
