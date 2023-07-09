import argparse
from pathlib import Path
import numpy as np
import copy
import pickle
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from datasets.vcoco import build as build_dataset
from models.backbone import build_backbone
from models.transformer import build_transformer, FeedForwardNetwork
from models.hoi import GroupWiseLinear
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


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
        return out


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


class PostProcessHOI(nn.Module):

    def __init__(self, num_queries, subject_category_id, correct_mat):
        super().__init__()
        self.max_hois = 100

        self.num_queries = num_queries
        self.subject_category_id = subject_category_id

        correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
        self.register_buffer('correct_mat', torch.from_numpy(correct_mat))

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
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(b.to('cpu').numpy(), l.to('cpu').numpy())]

            hoi_scores = vs * os.unsqueeze(1)

            verb_labels = torch.arange(hoi_scores.shape[1], device=self.correct_mat.device).view(1, -1).expand(
                hoi_scores.shape[0], -1)
            object_labels = ol.view(-1, 1).expand(-1, hoi_scores.shape[1])
            masks = self.correct_mat[verb_labels.reshape(-1), object_labels.reshape(-1)].view(hoi_scores.shape)
            hoi_scores *= masks

            ids = torch.arange(b.shape[0])

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(ids[:ids.shape[0] // 2].to('cpu').numpy(),
                                                                     ids[ids.shape[0] // 2:].to('cpu').numpy(),
                                                                     verb_labels.to('cpu').numpy(), hoi_scores.to('cpu').numpy())]

            results.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })

        return results


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--interaction_decoder', action='store_true',
                        help="Use interaction decoder")
    parser.add_argument('--image_verb_loss', action='store_true',
                        help="Use image-level binary classification loss for interaction")
    parser.add_argument('--x_improved', action='store_true',
                        help="Improve feature learning")
    parser.add_argument('--cat_specific_fc', action='store_true',
                        help="One fc for each category for projection")
    parser.add_argument('--verb_embed_norm', action='store_true',
                        help="Normalize w and x in verb cls")
    parser.add_argument('--i_dec_layers', default=1, type=int,
                        help="Number of decoding layers in the interaction decoder")
    parser.add_argument('--cross_attn_first', action='store_true',
                        help="In interaction decoder, use cross-attn before self-attn")
    parser.add_argument('--i_hidden_dim', default=256, type=int,
                        help="Size of the category query embeddings")

    # * HOI
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--missing_category_id', default=80, type=int)

    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--param_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)

    return parser


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                     37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                     82, 84, 85, 86, 87, 88, 89, 90)

    verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                    'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                    'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                    'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                    'point_instr', 'read_obj', 'snowboard_instr']

    device = torch.device(args.device)

    dataset_val = build_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    args.lr_backbone = 0
    args.masks = False
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    i_decoder, ffn = None, None
    if args.interaction_decoder:
        i_decoder = build_transformer(args, "i_decoder")
    if args.image_verb_loss:
        ffn = FeedForwardNetwork(
            d_model=args.i_hidden_dim,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
        )
    model = DETRHOI(
        backbone,
        transformer,
        num_obj_classes=len(valid_obj_ids) + 1,
        num_verb_classes=len(verb_classes),
        num_queries=args.num_queries,
        use_interaction_decoder=args.interaction_decoder,
        x_improved=args.x_improved,
        cat_specific_fc=args.cat_specific_fc,
        verb_embed_norm=args.verb_embed_norm,
        interaction_decoder=i_decoder,
        use_image_verb_loss=args.image_verb_loss,
        ffn=ffn,
    )
    post_processor = PostProcessHOI(args.num_queries, args.subject_category_id, dataset_val.correct_mat)
    model.to(device)
    post_processor.to(device)

    checkpoint = torch.load(args.param_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    detections = generate(model, post_processor, data_loader_val, device, verb_classes, args.missing_category_id)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(detections, f, protocol=2)


@torch.no_grad()
def generate(model, post_processor, data_loader, device, verb_classes, missing_category_id):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    detections = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = post_processor(outputs, orig_target_sizes)

        for img_results, img_targets in zip(results, targets):
            for hoi in img_results['hoi_prediction']:
                detection = {
                    'image_id': img_targets['img_id'],
                    'person_box': img_results['predictions'][hoi['subject_id']]['bbox'].tolist()
                }
                if img_results['predictions'][hoi['object_id']]['category_id'] == missing_category_id:
                    object_box = [np.nan, np.nan, np.nan, np.nan]
                else:
                    object_box = img_results['predictions'][hoi['object_id']]['bbox'].tolist()
                cut_agent = 0
                hit_agent = 0
                eat_agent = 0
                for idx, score in zip(hoi['category_id'], hoi['score']):
                    verb_class = verb_classes[idx]
                    score = score.item()
                    if len(verb_class.split('_')) == 1:
                        detection['{}_agent'.format(verb_class)] = score
                    elif 'cut_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        cut_agent = score if score > cut_agent else cut_agent
                    elif 'hit_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        hit_agent = score if score > hit_agent else hit_agent
                    elif 'eat_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        eat_agent = score if score > eat_agent else eat_agent
                    else:
                        detection[verb_class] = object_box + [score]
                        detection['{}_agent'.format(
                            verb_class.replace('_obj', '').replace('_instr', ''))] = score
                detection['cut_agent'] = cut_agent
                detection['hit_agent'] = hit_agent
                detection['eat_agent'] = eat_agent
                detections.append(detection)

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
