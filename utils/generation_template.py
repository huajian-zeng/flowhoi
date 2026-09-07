# Copyright (c) Meta Platforms, Inc. and affiliates.

from utils.parser_util import FullModelArgs


def get_template(args: FullModelArgs, guidance=False):
    if guidance:
        return guidance_template(args)
    return args


def guidance_template(args: FullModelArgs):
    args.do_inpaint = True
    args.gen_two_stages = True
    args.p2p_impute = True
    return args
