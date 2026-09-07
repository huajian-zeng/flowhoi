# Copyright (c) Meta Platforms, Inc. and affiliates.

from argparse import Namespace
import re
from os.path import join as pjoin


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, mode, data_repr, max_motion_length, use_abs3d=False,
            use_contacts=False, hands_only=False, text_detailed=False):
    """Load options, including EgoDex's 160-D axis-angle hand, SDF, object-pose, and articulation layout."""
    opt = Namespace()
    opt_dict = vars(opt)
    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 'grab':
        opt.data_root = './dataset/GRAB_HANDS'
        opt.motion_dir = pjoin(opt.data_root, data_repr)
        opt.joints_num = 42
        if hands_only:
            opt.dim_pose = 108
            opt.text_dir = pjoin(opt.data_root, 'texts_grasp')
        else:
            opt.dim_pose = 117
            if text_detailed:
                opt.text_dir = pjoin(opt.data_root, 'texts_detailed')
            else:
                opt.text_dir = pjoin(opt.data_root, 'texts_simple')
        if use_contacts:
            opt.dim_pose += 42
        opt.max_motion_length = 196
    elif opt.dataset_name == 'hot3d':
        opt.data_root = './dataset/HOT3D_HANDS'
        opt.motion_dir = pjoin(opt.data_root, data_repr)
        opt.joints_num = 42
        if hands_only:
            opt.dim_pose = 108
            opt.text_dir = pjoin(opt.data_root, 'texts_grasp')
        else:
            opt.dim_pose = 117
            opt.text_dir = pjoin(opt.data_root, 'texts_detailed')
        if use_contacts:
            opt.dim_pose += 42
        opt.max_motion_length = 200
    elif opt.dataset_name == 'egodex':
        opt.data_root = './dataset/EGODEX_HANDS'
        opt.motion_dir = pjoin(opt.data_root, data_repr)
        opt.joints_num = 42
        opt.dim_pose = 160
        opt.text_dir = pjoin(opt.data_root, 'texts_detailed')
        opt.max_motion_length = 576
    else:
        raise KeyError('Dataset not recognized')

    opt.num_classes = 200 // opt.unit_length
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt
