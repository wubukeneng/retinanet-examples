import os.path
import io
import numpy as np
import math
import torch
import torch.nn as nn

from . import backbones as backbones_mod
from ._C import Engine
from .box import generate_anchors, snap_to_anchors, decode, nms
from .box import generate_anchors_rotated, snap_to_anchors_rotated, nms_rotated
from .loss import FocalLoss, SmoothL1Loss

class Model(nn.Module):
    'RetinaNet - https://arxiv.org/abs/1708.02002'

    def __init__(self, backbones='ResNet50FPN', classes=80, 
                ratios=[1.0, 2.0, 0.5], scales=[4 * 2 ** (i / 3) for i in range(3)],
                angles=None, rotated_bbox=False, config={}, exporting=False, global=False, global_classes=2):
        super().__init__()

        if not isinstance(backbones, list):
            backbones = [backbones] 

        self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})
        self.name = 'RetinaNet'
        self.exporting = exporting
        self.rotated_bbox = rotated_bbox

        self.ratios = ratios
        self.scales = scales
        self.angles = angles if angles is not None else \
                    [-np.pi / 6, 0, np.pi / 6] if self.rotated_bbox else None
        self.anchors = {}
        self.classes = classes

        self.threshold = config.get('threshold', 0.05)
        self.top_n = config.get('top_n', 1000)
        self.nms = config.get('nms', 0.5)
        self.detections = config.get('detections', 100)

        self.stride = max([b.stride for _, b in self.backbones.items()])

        # classification and box regression heads
        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.num_anchors = self.num_anchors if not self.rotated_bbox else (self.num_anchors * len(self.angles))
        self.cls_head = make_head(classes * self.num_anchors)
        self.box_head = make_head(4 * self.num_anchors) if not self.rotated_bbox \
                        else make_head(6 * self.num_anchors)  # theta -> cos(theta), sin(theta)
        
        # global classification supervision
        self.global = global
        self.global_classes = global_classes
        
        if self.global:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.global_fc = nn.Linear(256, self.global_classes)
            
            self.global_criterion = FocalLoss()

        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss(beta=0.11)

    def __repr__(self):
        return '\n'.join([
            '     model: {}'.format(self.name),
            '  backbone: {}'.format(', '.join([k for k, _ in self.backbones.items()])),
            '   classes: {}, anchors: {}'.format(self.classes, self.num_anchors)
        ])

    def initialize(self, pre_trained):
        backbone_arr = list(self.backbones.keys())[0].rsplit('_', 1)
        mod = True if len(backbone_arr) > 1 else False
        
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            
            # TODO : ignore global fc
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            if self.rotated_bbox:
                ignored += ['box_head.8.bias', 'box_head.8.weight']
            
            if mod:
                weights = {}
                for k, v in chk['state_dict'].items():
                    if backbone_arr[0] in k:
                        k_arr = k.split(backbone_arr[0])
                        mod_k = '{}{}_Mod{}'.format(k_arr[0], backbone_arr[0], k_arr[1])
                        
                        if k not in ignored:
                            weights[mod_k] = v
            else:
                weights = {k: v for k, v in chk['state_dict'].items() if k not in ignored}
                
            state_dict.update(weights)
            self.load_state_dict(state_dict, strict=False)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbone(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)

            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

        # Initialize class head prior
        def initialize_prior(layer):
            pi = 0.01
            b = - math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)

        self.cls_head[-1].apply(initialize_prior)
        if self.rotated_bbox:
            self.box_head[-1].apply(initialize_prior)
        
        # we are using focal loss for global cls prediction
        self.global_fc.apply(initialize_prior)

    def forward(self, x, rotated_bbox=None):
        if self.training:
            if not self.global:
                x, targets = x
            else:
                x, targets, g_targets = x

        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))

        # Heads forward pass
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]
        
        # calc global head on p7
        if self.global:
            global_pools = [self.global_pool(t[-1]) for t in features]
            global_pools = [global_pool.view(-1, 256) for global_pool in global_pools]
            global_cls = [self.global_fc(p) for p in global_pools]

        if self.training:
            if self.global:
                return self._compute_loss(x, cls_heads, box_heads, targets.float(), global_cls=global_cls, g_targets=g_targets)
            else:
                return self._compute_loss(x, cls_heads, box_heads, targets.float())

        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]

        if self.exporting:
            self.strides = [x.shape[-1] // cls_head.shape[-1] for cls_head in cls_heads]
            return cls_heads, box_heads

        global nms, generate_anchors
        if self.rotated_bbox:
            nms = nms_rotated
            generate_anchors = generate_anchors_rotated

        # Inference post-processing
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            # Generate level's anchors
            stride = x.shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)

            # Decode and filter boxes
            decoded.append(decode(cls_head, box_head, stride, self.threshold, 
                                self.top_n, self.anchors[stride], self.rotated_bbox))

        # Perform non-maximum suppression
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, self.nms, self.detections)
    
    def post_processing(self, x, cls_heads, box_heads, threshold):
        global nms, generate_anchors
        if self.rotated_bbox:
            nms = nms_rotated
            generate_anchors = generate_anchors_rotated

        # Inference post-processing
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            # Generate level's anchors
            stride = x.shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)

            # Decode and filter boxes
            decoded.append(decode(cls_head, box_head, stride, threshold, 
                                self.top_n, self.anchors[stride], self.rotated_bbox))

        # Perform non-maximum suppression
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, self.nms, self.detections)

    def _extract_targets(self, targets, stride, size):
        global generate_anchors, snap_to_anchors
        if self.rotated_bbox:
            generate_anchors = generate_anchors_rotated
            snap_to_anchors = snap_to_anchors_rotated
        cls_target, box_target, depth = [], [], []
        for target in targets:
            target = target[target[:, -1] > -1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)
            
            anchors = self.anchors[stride]
            if not self.rotated_bbox:
                anchors = anchors.to(targets.device)
            snapped = snap_to_anchors(target, [s * stride for s in size[::-1]], stride, 
                                    anchors, self.classes, targets.device)
            for l, s in zip((cls_target, box_target, depth), snapped): l.append(s)
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def _compute_loss(self, x, cls_heads, box_heads, targets, global_cls=None, g_targets=None):
        cls_losses, box_losses, fg_targets = [], [], []
            
        for cls_head, box_head in zip(cls_heads, box_heads):
            size = cls_head.shape[-2:]
            stride = x.shape[-1] / cls_head.shape[-1]

            cls_target, box_target, depth = self._extract_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            cls_head = cls_head.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_head, cls_target)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            box_head = box_head.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_head, box_target)
            box_loss = box_mask * box_loss
            box_losses.append(box_loss.sum())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        
        if global_cls is not None:
            global_losses = []
            
            for global_head in global_cls:
                global_loss = self.global_criterion(global_head, g_targets)
                global_losses.append(global_loss.mean())
                
            global_loss = torch.stask(global_losses).mean()
            
            return cls_loss, box_loss, global_loss
        
        return cls_loss, box_loss

    def save(self, state):
        checkpoint = {
            'backbone': [k for k, _ in self.backbones.items()],
            'classes': self.classes,
            'state_dict': self.state_dict(),
            'ratios': self.ratios,
            'scales': self.scales
        }
        if self.rotated_bbox and self.angles:
            checkpoint['angles'] = self.angles

        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in state:
                checkpoint[key] = state[key]

        torch.save(checkpoint, '{}_{}.pth'.format(state['path'], state['iteration']))

    @classmethod
    def load(cls, filename, rotated_bbox=False, config={}, exporting=False):
        if not os.path.isfile(filename):
            raise ValueError('No checkpoint {}'.format(filename))

        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        kwargs = {}
        for i in ['ratios', 'scales', 'angles']:
            if i in checkpoint:
                kwargs[i] = checkpoint[i]
        if ('angles' in checkpoint) or rotated_bbox:
            kwargs['rotated_bbox'] = True
        # Recreate model from checkpoint instead of from individual backbones
        model = cls(backbones=checkpoint['backbone'], classes=checkpoint['classes'], config=config, exporting=exporting, **kwargs)
        model.load_state_dict(checkpoint['state_dict'])

        state = {}
        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in checkpoint:
                state[key] = checkpoint[key]

        del checkpoint
        torch.cuda.empty_cache()

        return model, state

    def export(self, size, batch, precision, calibration_files, calibration_table, verbose, onnx_only=False):

        import torch.onnx.symbolic_opset11 as onnx_symbolic
        def upsample_nearest2d(g, input, output_size, *args):
            # Currently, TRT 5.1/6.0/7.0 ONNX Parser does not support all ONNX ops
            # needed to support dynamic upsampling ONNX forumlation
            # Here we hardcode scale=2 as a temporary workaround
            scales = g.op("Constant", value_t=torch.tensor([1., 1., 2., 2.]))
            return g.op("Upsample", input, scales, mode_s="nearest")

        onnx_symbolic.upsample_nearest2d = upsample_nearest2d

        # Export to ONNX
        print('Exporting to ONNX...')
        self.exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros([1, 3, *size]).cuda()
        extra_args = {'opset_version': 11, 'verbose': verbose}
        torch.onnx.export(self.cuda(), zero_input, onnx_bytes, **extra_args)
        self.exporting = False

        if onnx_only:
            return onnx_bytes.getvalue()

        # Build TensorRT engine
        model_name = '_'.join([k for k, _ in self.backbones.items()])
        anchors = []
        if not self.rotated_bbox:
            anchors = [generate_anchors(stride, self.ratios, self.scales, 
                    self.angles).view(-1).tolist() for stride in self.strides]
        else:
            anchors = [generate_anchors_rotated(stride, self.ratios, self.scales, 
                    self.angles)[0].view(-1).tolist() for stride in self.strides]
        # Set batch_size = 1 batch/GPU for EXPLICIT_BATCH compatibility in TRT
        batch = 1
        return Engine(onnx_bytes.getvalue(), len(onnx_bytes.getvalue()), batch, precision,
                      self.threshold, self.top_n, anchors, self.rotated_bbox, self.nms, self.detections, 
                      calibration_files, model_name, calibration_table, verbose)
