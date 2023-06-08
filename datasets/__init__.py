# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .thumos_20220517 import build as build_thumos14
from .anet import build as build_activitynet
from .hacs import build as build_hacs
from .ucf101_update import build as build_ucf101
def build_dataset(image_set, args):
    if args.dataset_file == 'thumos14':
        return build_thumos14(image_set, args)
    elif args.dataset_file == 'activityNet':
        return build_activitynet(image_set, args)
    elif args.dataset_file == 'HACS':
        return build_hacs(image_set, args)
    elif args.dataset_file == 'ucf101':
        return build_ucf101(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
